from utils import *
from pathlib import Path
import re
import ast

class myCallback(TrainerCallback):
    def on_evaluate(self, state, args, control, metrics=None, logs=None, eval_dataloader=None, **kwargs):
        assert metrics["epoch"] >= getattr(self, "current_epoch", 0)
        if metrics["epoch"] > getattr(self, "current_epoch", 0):
            self.latest_acc = {}
            self.current_epoch = metrics["epoch"]
        for key in metrics.keys():
            if key.endswith("acc"):
                self.latest_acc[key] = metrics[key]
        if len(self.latest_acc) == len(length_ranges):
            valid_acc = self.latest_acc[f"eval_len{length_ranges[1][0]}-{length_ranges[1][1]}_acc"]
            global best_config, best_acc, highest_val_acc
            if valid_acc > highest_val_acc:
                best_config = config
                best_acc = deepcopy(self.latest_acc)
                highest_val_acc = valid_acc


parser = argparse.ArgumentParser()
parser.add_argument("--nope", action="store_true")
parser.add_argument("--diff_ij", type=int)
parser.add_argument("--length_range", type=str, default="small", choices=["small", "large"])
parser.add_argument("--vocab_factor", type=float, default=1.0)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

# fixed param (change to have different figures)
if args.length_range == "small":
    length_ranges = [(0, 50), (51, 100), (101, 150)]
elif args.length_range == "large":
    length_ranges = [(0, 128), (129, 256), (257, 384)]
max_test_length = length_ranges[-1][1]
diff_ij = args.diff_ij
vocab_size = int(max_test_length * args.vocab_factor)

path = f"{'APE' if not args.nope else 'NoPE'}-diff_ij_{diff_ij}-maxlength_{max_test_length}-vocab_{vocab_size}.txt"
if args.resume and Path(path).exists():
    log_f = open(path, "a+")
    log_f.seek(0)
    lines = log_f.readlines()
    start_idx = len(lines)
    last_line = lines[-1]
    best_config = ast.literal_eval(re.search(r"best config so far\:  (\{.+\}) =====", last_line).group(1))
    best_acc = ast.literal_eval(re.search(r"(\{\'eval.+\}) ===== best", last_line).group(1))
    highest_val_acc = float(re.search(r"(\d\.\d+) \{\'eval", last_line).group(1))
    assert highest_val_acc < 1.0
else:
    log_f = open(path, "w")
    start_idx = 0
    best_config = None
    best_acc = None
    highest_val_acc = 0

if not args.nope:
    search_space = dict(
        batch_size = [64,],
        lr = [1e-3, 1e-4],
        d_model = [64, 256, 512],
        n_layer = [2,],
        n_head = [1,],
        dropout = [0, 0.1],
    )
else:
     search_space = dict(
        batch_size = [64,],
        lr = [1e-3, 1e-4],
        d_model = [64, 256, 512],
        n_layer = [2, 3, 4, 5],
        n_head = [1, 2, 4],
        dropout = [0, 0.1],
    )

configs = make_configs(search_space)


for config in configs[start_idx:]:
    tokenizer = customTokenizer([str(i) for i in range(vocab_size)])
    train_dataset = UniqueCopyDataset(tokenizer, length_ranges[0], max_test_length, diff_ij)
    n_positions = train_dataset.determine_n_positions()

    test_dataset = {
        f"len{test_range[0]}-{test_range[1]}": EvalDataset(UniqueCopyDataset(tokenizer, test_range, -1, diff_ij), 5000)
            for test_range in length_ranges
    }

    per_device_bz = config.batch_size // torch.cuda.device_count() if torch.cuda.is_available() else config.batch_size 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(0)
    random.seed(0)

    cfg = GPT2Config(vocab_size=len(tokenizer), 
                    n_positions=n_positions,
                    n_embd=config.d_model,
                    n_layer=config.n_layer,
                    n_head=config.n_head,
                    bos_token_id=tokenizer.bos_token_id, 
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    attn_pdrop=0,
                    resid_pdrop=config.dropout,
                    embd_pdrop=config.dropout,
                    )

    if not args.nope:
        model = GPT2LMHeadModel(cfg)
    else:
        model = NoPEGPT2LMHeadModel(cfg)

    training_args = TrainingArguments(
        output_dir="./temp",    
        overwrite_output_dir=True,
        per_device_train_batch_size=per_device_bz,
        per_device_eval_batch_size=per_device_bz,
        max_steps=30_000,
        evaluation_strategy="steps",
        eval_steps=3_000,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=3_000,
        learning_rate=config.lr,
        weight_decay=0.01,
        optim='adamw_torch',
        lr_scheduler_type='linear',
        warmup_steps=1500,
        report_to="none",
    )

    data_collator = customCollator(tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[myCallback],
    )

    trainer.train()


    print(highest_val_acc, best_acc, "===== best config so far: ", best_config, "===== current config: ", config, file=log_f)
    log_f.flush()

    if highest_val_acc == 1.0:
        break

log_f.close()