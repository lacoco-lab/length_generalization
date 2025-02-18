import torch
import wandb
import hydra
import logging
from tqdm import tqdm
from typing import Optional

from dataset_utils import create_dataloader, build_lang_config

from omegaconf import DictConfig, OmegaConf
from config import settings

from transformers import GPT2LMHeadModel, GPT2Config

logger = logging.getLogger(__name__)


class NoPE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return 0

class NoPEGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer.wpe = NoPE()


class RegGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, coef):
        super().__init__(config)
        self.coef = coef
    
    def forward(self, *args, labels: Optional[torch.LongTensor] = None, **kwargs):
        outputs = super().forward(*args, labels=labels, **kwargs)
        if labels is not None:
            loss2 = self.compute_regularizer()
            # print("computed regulariser")
            if isinstance(outputs, tuple):
                outputs = (outputs[0] + loss2 * self.coef,) + outputs[1:]
            else:
                outputs.loss = outputs.loss + loss2 * self.coef
        return outputs
    
    def compute_regularizer(self):
        pe = self.transformer.wpe.weight # (num_embeddings, embedding_dim)
        square_sum = 0
        for block in self.transformer.h:
            w_matrix = block.attn.c_attn.weight # W_qkv for this layer (including all heads), 
            # it can first be split (by columns) into 3 equal part, correspond to q, k, v. Each part then be spit into many parts for each head
            k_offset = block.attn.embed_dim
            head_dim = block.attn.head_dim
            for i in range(block.attn.num_heads):
                w_query = w_matrix[:, i*head_dim : (i+1)*head_dim]  # W_q for head i
                w_key = w_matrix[:, k_offset+i*head_dim : k_offset+(i+1)*head_dim]  # W_k for head i
                product = (pe @ w_query) @ ((pe @ w_key).T)
                product = (torch.tril(product)**2).sum(dim=0).mean()
                square_sum = square_sum + product
        return square_sum


def get_model(cfg, max_seq_length, vocab_size, encoder, bos_token, eos_token, pad_token, device):
    # Add these parameters to the hydra config
    n_emb, n_layer, n_head = cfg.model.d_model, cfg.model.num_layers, cfg.model.num_heads
    use_nope, use_reg, reg_coef = cfg.model.use_nope, cfg.model.use_reg, cfg.model.reg_coef

    # print(vocab_size, max_seq_length, n_emb, n_layer, n_head, encoder[bos_token], encoder[eos_token], encoder[pad_token])
    # print(type(vocab_size), type(max_seq_length), type(n_emb), type(n_layer), type(n_head), type(encoder[bos_token]), type(encoder[eos_token]), type(encoder[pad_token]))
    cfg = GPT2Config(
                vocab_size=vocab_size,
                n_positions=2*max_seq_length,
                n_embd=n_emb,
                n_layer=n_layer,
                n_head=n_head,
                bos_token_id=encoder[bos_token].item(),
                eos_token_id=encoder[eos_token].item(),
                pad_token_id=encoder[pad_token].item(),
                attn_pdrop=0,
                resid_pdrop=0,
                embd_pdrop=0
            )

    if use_reg:
        model = RegGPT2LMHeadModel(cfg, reg_coef)
    elif use_nope:
        model = NoPEGPT2LMHeadModel(cfg)
    else:
        model = GPT2LMHeadModel(cfg)

    model.to(device)
    # print(model)
    return model


def compute_loss_with_padding_ignore(logits, labels, pad_token_id, loss_fn, logger):
    """
    logits: (batch_size, seq_length, vocab_size) - output of the model
    labels: (batch_size, seq_length) - true labels
    pad_token_id: int - id of the padding token to be ignored in the loss calculation
    """
    # Create a mask to ignore pad tokens in the labels
    pad_mask = labels != pad_token_id  # (batch_size, seq_length)

    # Flatten the logits and labels to compute the loss
    logits_flat = logits.view(-1, logits.size(-1))  # (batch_size * seq_length, vocab_size)
    labels_flat = labels.view(-1)  # (batch_size * seq_length)

    loss = loss_fn(logits_flat, labels_flat)  # (batch_size * seq_length)

    # Reshape the loss back to (batch_size, seq_length)
    loss = loss.view(labels.size())

    # Zero-out losses for pad tokens
    loss = loss * pad_mask  # (batch_size, seq_length)

    # Return the mean loss only for non-pad tokens
    num_non_pad_tokens = pad_mask.sum().item()
    mean_loss = loss.sum() / num_non_pad_tokens if num_non_pad_tokens > 0 else loss.sum()

    predictions = torch.argmax(logits, dim=-1)  # (batch_size, seq_length)
    logger.info("Predictions {} Labels {}".format(predictions[0], labels[0]))
    correct_predictions = (predictions == labels) & pad_mask  # ignore pad tokens

    # Accuracy is the number of correct predictions divided by the number of non-pad tokens
    correct_predictions = correct_predictions.all(dim=-1)  # (batch_size)
    # Accuracy is the number of correct predictions divided by the number of non-pad tokens
    batch_accuracy = correct_predictions.float().mean().item()    
    return loss.mean(), mean_loss, batch_accuracy

def offset_and_forward(model, inputs, labels, use_reg=False):
    # If no position_ids are provided, we will create them with a shift of k
    batch_size, seq_length = inputs.shape
    # Generate a random shift k for each sample in the batch
    # Adjust the range of k
    high_start = int(model.config.n_positions/2)
    shift_k = torch.randint(high_start, (batch_size,)).to(inputs.device)
    # Create position_ids with the shift applied
    position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
    position_ids = position_ids + shift_k.unsqueeze(1)
    if use_reg:
        return model(inputs, position_ids=position_ids, labels=labels)
    else: 
        return model(inputs, position_ids=position_ids)


def train_with_ce(model, vocab_size, pad_token_id, loss_fn, device, epochs, optimizer, dataloader_dict, use_reg, use_wandb, logger):
    """
    Args:
        model (torch.nn.Module): Model with / without NOPE
        loss_fn (torch.nn.MSELoss): Loss function
        device (torch.device): The GPU device on which to run the pipeline
        epochs (int): Number of epochs to train for
        optimizer (torch.optim.Optimizer): Pytorch Optimizer, No it's loscurrently AdamW
        dataloader_dict (dict): Key - train, or val_bin_number, and the values are the corresponding dataloaders
    """
    model.train()
    for epoch in tqdm(range(int(epochs))):
        logging_dict = {}
        for category, dataloader in dataloader_dict.items():
            if category == 'train':
                model.train()
            else:
                # Go for validation sets, once training is about to finish
                model.eval()

            epoch_loss, epoch_accuracy, num_batches = 0, 0, 0
            for data in dataloader:
                inputs, targets = data['input'], data['output']
                inputs = inputs.to(device).long()
                targets = targets.to(device).long()
                # Print the maximum value in the input tensor
                if category == 'train':
                    predicted = offset_and_forward(model, inputs, targets, use_reg=use_reg)
                else:
                    predicted = model(inputs)
                predicted = predicted.logits

                loss, batch_mean_loss, batch_accuracy = compute_loss_with_padding_ignore(predicted, targets, pad_token_id, loss_fn, logger)
                # Since each batch size is the same, we can just sum the batch accuracies ; and divide by #batches
                epoch_accuracy += batch_accuracy
                num_batches += 1
                epoch_loss += batch_mean_loss

                if category == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    # Optimizer Step and scheduler step
                    optimizer.step()
            
            logging_dict = {
                f'{category}_Loss': epoch_loss,
                f'{category}_Accuracy': epoch_accuracy / num_batches
            }
            if use_wandb:
                wandb.log(logging_dict)
            else:
                logger.info(logging_dict)


@hydra.main(config_path='configs', config_name="defaults", version_base=None)
def run_pipeline(cfg: DictConfig) -> None:

    use_wandb = cfg.basic.use_wandb
    logger = logging.getLogger()
    if use_wandb is True:
        # Login into the wandb system
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        wandb.login(key=settings.WANDB_API_KEY)
        name = cfg.dataset.name + '_nope:' + str(cfg.model.use_nope) + '_l:' + str(cfg.model.num_layers) + '_h:' + str(cfg.model.num_heads)
        wandb.init(project='length_generalization', entity=settings.WANDB_TEAM,
                   name=name, config=cfg_copy)    

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(" DEVICE & LOSSFN DONE ")
    
    # TO BE ADDED TO CONFIG
    bos_token, eos_token  = cfg.basic.bos_token, cfg.basic.eos_token
    pad_token, num_val_bins = cfg.basic.pad_token, cfg.dataset.num_val_bins
    max_seq_size, generate_dataset = cfg.basic.max_seq_size, cfg.basic.generate_dataset

    dataset_name, batch_size, model_name = cfg.dataset.name, cfg.dataset.batch_size, cfg.model.name

    # If required, build the language config from the hydra dataset config
    lang_params = build_lang_config(cfg.dataset) if generate_dataset else None

    # Need alternate dataloader here for cross entropy
    dataloader_dict, dataset, max_seq_length = create_dataloader(
        base_folder=dataset_name,
        batch_size=batch_size,
        lang_params=lang_params,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token, 
        max_size=max_seq_size, 
        num_val_bins=num_val_bins,
        generate=generate_dataset, 
    )
    print("Dataloader done")

    # To account for padding token 
    vocab_size, encoder = len(dataset.vocab) + 1, dataset.encoder
    #print(vocab_size_in, vocab_size_out)

    model = get_model(cfg, max_seq_length, vocab_size, encoder, bos_token, eos_token, pad_token, device)
    # The learning rate is added to the optimizer by default, model parameters added manually
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    num_epochs = cfg.train.epochs
    print(" STARTING TRAINING ")

    # Add argument to train_model
    train_with_ce(
        model=model, 
        vocab_size=vocab_size,
        pad_token_id=encoder[pad_token].item(),
        loss_fn=loss_fn, 
        device=device, 
        epochs=num_epochs, 
        optimizer=optimizer, 
        dataloader_dict=dataloader_dict, 
        use_reg=cfg.model.use_reg,
        use_wandb=use_wandb,
        logger=logger
    )
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    run_pipeline()
