from utils import *
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
            global run_highest_val_acc, run_best_acc
            if valid_acc > run_highest_val_acc:
                run_best_acc = deepcopy(self.latest_acc)
                run_highest_val_acc = valid_acc


parser = argparse.ArgumentParser()
parser.add_argument("--nope", action="store_true")
parser.add_argument("--diff_ij", type=int)
parser.add_argument("--length_range", type=str, default="small", choices=["small", "large"])
parser.add_argument("--vocab_factor", type=float, default=1.0)
parser.add_argument("--num_run", type=int, default=5)
args = parser.parse_args()

if args.length_range == "small":
    length_ranges = [(0, 50), (51, 100), (101, 150)]
elif args.length_range == "large":
    length_ranges = [(0, 128), (129, 256), (257, 384)]
max_test_length = length_ranges[-1][1]
diff_ij = args.diff_ij
vocab_size = int(max_test_length * args.vocab_factor)

path = f"{'APE' if not args.nope else 'NoPE'}-diff_ij_{diff_ij}-maxlength_{max_test_length}-vocab_{vocab_size}.txt"

with open(path, "r") as f:
    last_line = f.readlines()[-1]
config = ast.literal_eval(re.search(r"best config so far\:  (\{.+\}) =====", last_line).group(1))
config = EasyDict(config)

output_path = "multi_seeds-" + path
summary_f = open(output_path, "w")
print("hyper-parameters", config, file=summary_f)
summary_f.flush()


threshold = 0.99

results = {f"eval_len{test_range[0]}-{test_range[1]}_acc": [] for test_range in length_ranges}
for seed in range(1000):
    run_highest_val_acc = -1
    run_best_acc = None

    tokenizer = customTokenizer([str(i) for i in range(vocab_size)])
    train_dataset = UniqueCopyDataset(tokenizer, length_ranges[0], max_test_length, diff_ij)
    n_positions = train_dataset.determine_n_positions()

    test_dataset = {
        f"len{test_range[0]}-{test_range[1]}": EvalDataset(UniqueCopyDataset(tokenizer, test_range, -1, diff_ij), 5000)
            for test_range in length_ranges
    }

    per_device_bz = config.batch_size // torch.cuda.device_count() if torch.cuda.is_available() else config.batch_size 
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(seed)
    random.seed(seed)

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

    if run_best_acc[f"eval_len{length_ranges[0][0]}-{length_ranges[0][1]}_acc"] >= threshold:
        for key, value in run_best_acc.items():
            results[key].append(value)
        print("** ", run_best_acc, file=summary_f)
    else:
        print(run_best_acc, file=summary_f)
    summary_f.flush()

    if len(results[f"eval_len{length_ranges[0][0]}-{length_ranges[0][1]}_acc"]) == args.num_run:
        break
        
print("mean results\t\t",  "\t\t".join([f"{key}: {(sum(value) / args.num_run):.4f}" for key, value in results.items()]), file=summary_f)
summary_f.flush()

summary_f.close()
