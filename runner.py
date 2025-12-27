import argparse
import os
import time
import platform
import torch
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from experiments.prepare_datasets import get_dataloaders
from experiments.train import get_trainer
from models.config import MODEL_CONFIGS, TRAINING_CONFIGS, ARCHITECTURES
from models.transformer import KATClassifier, KATLanguageModelling


# Utility Functions
def get_system_info():
	"""Collect hardware/software info for reproducibility."""
	info = {"pytorch": torch.__version__, "python": platform.python_version()}
	if torch.cuda.is_available():
		info.update({
			"gpu": torch.cuda.get_device_name(0),
			"vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
			"cuda": torch.version.cuda,
		})
	return info


def count_params(model):
	"""Count parameters by module type (embedding, attention, ffn, other)."""
	total = sum(p.numel() for p in model.parameters())
	by_type = {"embed": 0, "attn": 0, "ffn": 0, "other": 0}

	for name, p in model.named_parameters():
		if "embedding" in name:
			by_type["embed"] += p.numel()
		elif "attn" in name:
			by_type["attn"] += p.numel()
		elif "ffn" in name:
			by_type["ffn"] += p.numel()
		else:
			by_type["other"] += p.numel()

	return {"total_params": total, **{f"params_{k}": v for k, v in by_type.items()}}


def gpu_mem():
	"""Get current GPU memory stats. Returns empty dict if no CUDA."""
	if not torch.cuda.is_available():
		return {}
	return {
		"mem_alloc_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
		"mem_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
	}

def measure_inference(model, vocab_size, device, n_runs=100):
	"""Measure single-sample inference latency in milliseconds."""
	model.eval()
	x = torch.randint(0, vocab_size, (1, 128)).to(device)

	# Warmup
	with torch.no_grad():
		for _ in range(10):
			model(x)

	# Synchronize before timing
	if torch.cuda.is_available():
		torch.cuda.synchronize()

	# Timed runs
	t = time.perf_counter()
	with torch.no_grad():
		for _ in range(n_runs):
			model(x)
	if torch.cuda.is_available():
		torch.cuda.synchronize()

	model.train()
	return round((time.perf_counter() - t) / n_runs * 1000, 3)

def setup_ddp():
	# read env from torchrun
	dist.init_process_group(backend="nccl")
	local_rank = int(os.environ["LOCAL_RANK"])
	torch.cuda.set_device(local_rank)
	return local_rank

def cleanup_ddp():
	dist.destroy_process_group()

def run_experiment(args):
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.reset_peak_memory_stats()
		torch.backends.cudnn.benchmark = True
		torch.set_float32_matmul_precision('high')

	# Config Override if batch is specified
	train_config = TRAINING_CONFIGS[args.task].copy()
	if args.batch_size:
		train_config["batch_size"] = args.batch_size
	if args.mini_batch_size:
		train_config["mini_batch_size"] = args.mini_batch_size
	# ------------------------------------------------------

	# DDP setup
	is_ddp = "LOCAL_RANK" in os.environ
	if is_ddp:
		local_rank = setup_ddp()
		device = torch.device(f"cuda:{local_rank}")
		# logging only master
		is_master = (local_rank == 0)
	else:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		is_master = True

	if is_master:
		print(f"Using device: {device}")
		print(f"TASK: {args.task}")
		print(f"--- Experiment: {args.arch} ({args.size}) | Seed: {args.seed} ---")
		print(f"Config: Batch={train_config['batch_size']} | Mini={train_config['mini_batch_size']}")

	# Config
	model_cfg = MODEL_CONFIGS[args.size]
	lr = train_config.get(f"lr_{args.arch}", 3e-4)

	# Checkpoints
	os.makedirs("checkpoints", exist_ok=True)
	ckpt_path = f"checkpoints/{args.task}_{args.arch}_{args.size}_s{args.seed}.pt"

	# Load Datasets
	train_loader, val_loader, test_loader, vocab_size, task_info = get_dataloaders(
		task=args.task,
		batch_size=train_config["mini_batch_size"],
		seq_len=args.seq_len,
		lm_dataset=args.lm_dataset,
		is_ddp=is_ddp
	)

	# Init Model
	if args.task == "classification":
		model = KATClassifier(
			vocab_size=vocab_size,
			num_classes=task_info["num_classes"],
			d_model=model_cfg["d_model"],
			n_heads=model_cfg["n_heads"],
			n_layers=model_cfg["n_layers"],
			n_hidden=model_cfg["n_hidden"],
			ffn_type=args.arch,
			d_ff=model_cfg["d_ff"][args.arch],
			dropout=model_cfg["dropout"],
			max_seq_len=512
		).to(device)

		primary_metric = "val_acc"
		metric_mode = "max"

	elif args.task == "language_modelling":
		model = KATLanguageModelling(
			vocab_size=vocab_size,
			d_model=model_cfg["d_model"],
			n_heads=model_cfg["n_heads"],
			n_layers=model_cfg["n_layers"],
			n_hidden=model_cfg["n_hidden"],
			ffn_type=args.arch,
			d_ff=model_cfg["d_ff"][args.arch],
			dropout=model_cfg["dropout"],
			max_seq_len=args.seq_len
		).to(device)

		primary_metric = "val_ppl"
		metric_mode = "min"

	# Compile before DDP
	if args.compile:
		if is_master: print("Compiling model...")
		model = torch.compile(model)

	# Wrap Model in DDP
	if is_ddp:
		model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

	if is_master:
		param_stats = count_params(model)
		print(f"Total params: {param_stats['total_params']:,} | FFN params: {param_stats['params_ffn']:,}")

		# Init Wandb only on master
		wandb.init(
			project="kan-transformer-thesis",
			name=f"{args.task}_{args.arch}_{args.size}_s{args.seed}",
			tags=[args.task, args.arch, args.size, task_info["dataset"]],
			config={
				"task": args.task,
				"dataset": task_info["dataset"],
				"arch": args.arch,
				"size": args.size,
				"seed": args.seed,
				"seq_len": args.seq_len,
				"d_model": model_cfg["d_model"],
				"n_heads": model_cfg["n_heads"],
				"n_layers": model_cfg["n_layers"],
				"n_hidden": model_cfg["n_hidden"],
				"d_ff": model_cfg["d_ff"][args.arch],
				"dropout": model_cfg["dropout"],
				"lr": lr,
				"batch_size": train_config["batch_size"],
				"mini_batch_size": train_config["mini_batch_size"],
				"grad_accum": train_config["batch_size"] // train_config["mini_batch_size"],
				**param_stats,
				**get_system_info(),
			}
		)

	# Trainer
	trainer = get_trainer(
		args.task, model, train_loader, val_loader, lr, device, train_config
	)

	# Early Stopping Setup
	if metric_mode == "max":
		best_metric = 0.0
		is_better = lambda new, old: new > old + train_config["early_stopping_min_delta"]
	else:
		best_metric = float('inf')
		is_better = lambda new, old: new < old - train_config["early_stopping_min_delta"]

	no_improve = 0
	best_epoch = 0
	best_f1 = 0.0
	patience = train_config["early_stopping_patience"]
	t_start = time.time()

	for epoch in range(train_config["max_epochs"]):
		# Sampler needs to know epoch
		if is_ddp and hasattr(train_loader.sampler, 'set_epoch'):
			train_loader.sampler.set_epoch(epoch)

		# Training
		if args.task == "classification":
			train_loss, train_acc, t_epoch = trainer.train_epoch(
				epoch,
				train_config["mini_batch_size"],
				train_config["batch_size"]
			)
		else:
			train_loss, train_ppl, t_epoch = trainer.train_epoch(
				epoch,
				train_config["mini_batch_size"],
				train_config["batch_size"]
			)

		# Evaluation
		if args.task == "classification":
			val_loss, val_acc, val_f1 = trainer.evaluate()
			current_metric = val_acc

			if is_master:
				wandb.log({
					"epoch": epoch + 1,
					"val/loss": val_loss,
					"val/acc": val_acc,
					"val/f1": val_f1,
					"train/epoch_avg_loss": train_loss,
					"train/epoch_avg_acc": train_acc,
					"epoch_time": t_epoch,
					**gpu_mem(),
				}, step=trainer.global_step)
				print(f"E{epoch + 1}: val_acc={val_acc:.2f}% | F1={val_f1:.2f}% | loss={val_loss:.4f} | "
				      f"t={t_epoch:.1f}s")

		else:  # language modelling
			val_loss, val_ppl = trainer.evaluate()
			current_metric = val_ppl

			if is_master:
				wandb.log({
					"epoch": epoch + 1,
					"val/loss": val_loss,
					"val/ppl": val_ppl,
					"train/epoch_avg_loss": train_loss,
					"train/epoch_avg_ppl": train_ppl,
					"epoch_time": t_epoch,
					**gpu_mem(),
				}, step=trainer.global_step)
				print(f"E{epoch + 1}: val_ppl={val_ppl:.2f} | loss={val_loss:.4f} | "
				      f"t={t_epoch:.1f}s")

		# Early Stopping
		if is_better(current_metric, best_metric):
			best_metric = current_metric
			best_epoch = epoch + 1
			if args.task == "classification":
				best_f1 = val_f1
			no_improve = 0

			if is_master:
				# Save unwrapped state_dict
				state_dict = model.module.state_dict() if is_ddp else model.state_dict()
				torch.save(state_dict, ckpt_path)
				metric_name = "acc" if args.task == "classification" else "ppl"
				print(f"  -> Saved new best model ({metric_name}={best_metric:.2f})")
		else:
			no_improve += 1
			if no_improve >= patience:
				if is_master: print(f"Early stopping at epoch {epoch + 1}")
				break

	# Final Evaluation (Only on Master to avoid double logging/printing)
	if is_master:
		total_time = (time.time() - t_start) / 60
		final_mem = gpu_mem()

		print("\nLoading best checkpoint for final evaluation...")

		# Load into unwrapped model or handle keys
		if is_ddp:
			model.module.load_state_dict(torch.load(ckpt_path, weights_only=True))
		else:
			model.load_state_dict(torch.load(ckpt_path, weights_only=True))

		inference_ms = measure_inference(model, vocab_size, device)

		# Test Evaluation
		if args.task == "classification":
			_, val_acc_verified, val_f1_verified = trainer.evaluate()
			test_loss, test_acc, test_f1 = trainer.evaluate(test_loader)

			wandb.run.summary.update({
				"best_val_acc": best_metric,
				"best_val_f1": best_f1,
				"test_acc": test_acc,
				"test_f1": test_f1,
				"best_epoch": best_epoch,
				"total_epochs": epoch + 1,
				"inference_ms": inference_ms,
				"total_time_min": total_time,
				"ffn_params": param_stats["params_ffn"],
				"params_non_embed": param_stats["total_params"] - param_stats["params_embed"],
				"peak_mem_gb": final_mem.get("mem_peak_gb", 0),
			})

			print(f"\nTest Acc: {test_acc:.2f}% | Test F1: {test_f1:.2f}%")

		else:  # language modelling
			_, val_ppl_verified = trainer.evaluate()
			test_loss, test_ppl = trainer.evaluate(test_loader)

			wandb.run.summary.update({
				"best_val_ppl": best_metric,
				"test_ppl": test_ppl,
				"best_epoch": best_epoch,
				"total_epochs": epoch + 1,
				"inference_ms": inference_ms,
				"total_time_min": total_time,
				"ffn_params": param_stats["params_ffn"],
				"params_non_embed": param_stats["total_params"] - param_stats["params_embed"],
				"peak_mem_gb": final_mem.get("mem_peak_gb", 0),
			})

			print(f"\nTest PPL: {test_ppl:.2f}")

		wandb.finish()

		print(f"\n{'=' * 50}")
		print(f"Experiment Complete: {args.task} | {args.arch} ({args.size})")
		if args.task == "classification":
			print(f"Best Val Acc:   {best_metric:.2f}%")
			print(f"Test Acc:       {test_acc:.2f}%")
			print(f"Test F1:        {test_f1:.2f}%")
		else:
			print(f"Best Val PPL:   {best_metric:.2f}")
			print(f"Test PPL:       {test_ppl:.2f}")
		print(f"Best Epoch:     {best_epoch}")
		print(f"Inference:      {inference_ms:.2f} ms")
		print(f"Peak VRAM:      {final_mem.get('mem_peak_gb', 0):.2f} GB")
		print(f"Total Time:     {total_time:.1f} min")
		print(f"{'=' * 50}")

	# Clean up DDP group
	if is_ddp:
		cleanup_ddp()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--task", choices=["classification", "language_modelling"],
	                    default="classification")
	parser.add_argument("--arch", type=str, default="mlp",
	                    choices=ARCHITECTURES,
	                    help="FFN architecture type")
	parser.add_argument("--size", type=str, default="tiny",
	                    choices=MODEL_CONFIGS.keys(),
	                    help="Model size config")
	parser.add_argument("--seed", type=int, default=42,
	                    help="Random seed for reproducibility")
	parser.add_argument("--seq_len", type=int, default=256,
	                    help="Sequence length for LM")
	parser.add_argument("--lm_dataset", choices=["fineweb", "wikitext"],
	                    default="fineweb", help="Dataset for language modelling")
	parser.add_argument("--compile", action="store_true",
	                    help="Use torch.compile")
	parser.add_argument("--batch_size", type=int, default=None,
	                    help="Override config batch size")
	parser.add_argument("--mini_batch_size", type=int, default=None,
	                    help="Override config mini batch size")

	args = parser.parse_args()
	run_experiment(args)