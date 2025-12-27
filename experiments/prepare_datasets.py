import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

CLASSIFICATION_CONFIG = {
	"ag_news": {
		"hf_path": "ag_news",
		"text_field": "text",
		"label_field": "label",
		"num_classes": 4,
	}
}


def prepare_classification_dataset(dataset_name="ag_news",
                                   save_dir=None,
                                   max_length=512):
	"""Prepare classification dataset"""

	config = CLASSIFICATION_CONFIG[dataset_name]

	if save_dir is None:
		save_dir = f"./data/processed_{dataset_name}"

	print(f"1. Loading tokenizer (GPT-2)...")
	tokenizer = AutoTokenizer.from_pretrained("gpt2")
	tokenizer.pad_token = tokenizer.eos_token

	print(f"2. Loading dataset {dataset_name}...")
	dataset = load_dataset(config["hf_path"])

	def tokenize_function(examples):
		return tokenizer(
			examples["text"],
			truncation=True,
			max_length=max_length
		)

	print("3. Tokenizing dataset...")
	tokenized_datasets = dataset.map(tokenize_function, batched=True)

	# Remove unnecessary columns
	cols_to_remove = [c for c in tokenized_datasets["train"].column_names
	                  if c not in ["input_ids", "attention_mask", "label"]]
	tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)

	# Rename 'label' to 'labels' for consistency
	tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
	tokenized_datasets.set_format("torch")

	print(f"4. Saving to {save_dir}...")
	os.makedirs(save_dir, exist_ok=True)
	tokenized_datasets.save_to_disk(save_dir)
	tokenizer.save_pretrained(save_dir)

	# Save metadata
	torch.save({
		"dataset_name": dataset_name,
		"num_classes": config["num_classes"],
		"max_length": max_length,
		"train_size": len(tokenized_datasets["train"]),
		"test_size": len(tokenized_datasets["test"]),
	}, os.path.join(save_dir, "metadata.pt"))

	print(f"Train: {len(tokenized_datasets['train']):,} samples")
	print(f"Test:  {len(tokenized_datasets['test']):,} samples")
	print("Done!")


def get_classification_dataloaders(dataset_name="ag_news",
                                   data_dir=None,
                                   batch_size=64,
                                   val_split=0.1,
                                   is_ddp=False):
	"""Load classification dataloaders with train/val/test split"""

	if data_dir is None:
		data_dir = f"./data/processed_{dataset_name}"

	dataset = load_from_disk(data_dir)
	tokenizer = AutoTokenizer.from_pretrained(data_dir)
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
	metadata = torch.load(os.path.join(data_dir, "metadata.pt"))

	# Split train into train/val
	train_val = dataset["train"].train_test_split(test_size=val_split, seed=42)

	# DDP
	if is_ddp:
		train_sampler = DistributedSampler(train_val["train"])
		shuffle = False
	else:
		train_sampler = None
		shuffle = True

	train_loader = DataLoader(
		train_val["train"],
		batch_size=batch_size,
		shuffle=shuffle,
		sampler=train_sampler,
		collate_fn=data_collator,
		num_workers=4,
		pin_memory=True
	)
	val_loader = DataLoader(
		train_val["test"],
		batch_size=batch_size,
		collate_fn=data_collator,
		num_workers=4,
		pin_memory=True
	)
	test_loader = DataLoader(
		dataset["test"],
		batch_size=batch_size,
		collate_fn=data_collator,
		num_workers=4,
		pin_memory=True
	)

	return train_loader, val_loader, test_loader, tokenizer.vocab_size, metadata


class LMDataset(Dataset):
	"""Chunked dataset for language modelling"""
	def __init__(self, tokens, seq_len):
		self.seq_len = seq_len
		n_chunks = len(tokens) // (seq_len + 1)
		self.chunks = tokens[:n_chunks * (seq_len + 1)].view(n_chunks, seq_len + 1)

	def __len__(self):
		return len(self.chunks)

	def __getitem__(self, idx):
		chunk = self.chunks[idx]
		return {
			"input_ids": chunk[:-1].clone(),
			"targets": chunk[1:].clone()
		}


def prepare_wikitext(save_dir="./data/processed_wikitext2", seq_len=256):
	"""Tokenize WikiText-2"""
	print("1. Loading tokenizer (GPT-2)...")
	tokenizer = AutoTokenizer.from_pretrained("gpt2")

	print("2. Loading WikiText-2...")
	dataset = load_dataset("wikitext", "wikitext-2-v1")

	os.makedirs(save_dir, exist_ok=True)

	for split in ["train", "validation", "test"]:
		print(f"3. Processing {split}...")

		texts = [t for t in dataset[split]["text"] if t.strip()]
		full_text = "\n".join(texts)
		tokens = tokenizer.encode(full_text, return_tensors="pt").squeeze(0)

		print(f"   {split}: {len(tokens):,} tokens")
		torch.save(tokens, os.path.join(save_dir, f"{split}_tokens.pt"))

	tokenizer.save_pretrained(save_dir)
	torch.save({"seq_len": seq_len, "vocab_size": tokenizer.vocab_size},
	           os.path.join(save_dir, "metadata.pt"))
	print("Done!")


def get_lm_dataloaders(data_dir, batch_size=64, seq_len=256, is_ddp=False):
	train_tokens = torch.load(os.path.join(data_dir, "train_tokens.pt"))
	val_tokens = torch.load(os.path.join(data_dir, "validation_tokens.pt"))
	test_tokens = torch.load(os.path.join(data_dir, "test_tokens.pt"))

	metadata = torch.load(os.path.join(data_dir, "metadata.pt"))

	train_dataset = LMDataset(train_tokens, seq_len)
	val_dataset = LMDataset(val_tokens, seq_len)
	test_dataset = LMDataset(test_tokens, seq_len)

	print(f"Train chunks: {len(train_dataset):,}")
	print(f"Val chunks: {len(val_dataset):,}")
	print(f"Test chunks: {len(test_dataset):,}")

	# DDP
	if is_ddp:
		train_sampler = DistributedSampler(train_dataset)
		shuffle = False
	else:
		train_sampler = None
		shuffle = True

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		sampler=train_sampler,
		drop_last=True,
		num_workers=4,
		pin_memory=True
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		drop_last=True,
		num_workers=4,
		pin_memory=True
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=batch_size,
		drop_last=True,
		num_workers=4,
		pin_memory=True
	)

	return train_loader, val_loader, test_loader, metadata["vocab_size"]


def prepare_fineweb(save_dir="./data/processed_fineweb",
                    target_tokens=30_000_000,
                    seq_len=256,
                    seed=42):
	print(f"1. Loading tokenizer (GPT-2)...")
	tokenizer = AutoTokenizer.from_pretrained("gpt2")

	print(f"2. Streaming FineWeb (target: {target_tokens:,} tokens)...")
	dataset = load_dataset(
		"HuggingFaceFW/fineweb",
		name="sample-10BT",
		split="train",
		streaming=True
	)

	dataset = dataset.shuffle(seed=seed, buffer_size=10000)

	os.makedirs(save_dir, exist_ok=True)

	all_tokens = []
	doc_count = 0

	for doc in dataset:
		text = doc.get("text", "")
		if not text.strip():
			continue

		tokens = tokenizer.encode(text, add_special_tokens=False)
		all_tokens.extend(tokens)
		doc_count += 1

		if len(all_tokens) >= target_tokens:
			break

		if doc_count % 1000 == 0:
			print(f"   {len(all_tokens):,} tokens from {doc_count:,} docs...", end="\r")

	print(f"   Final: {len(all_tokens):,} tokens from {doc_count:,} documents")

	# 90/5/5 Split
	n = len(all_tokens)
	train_end = int(n * 0.90)
	val_end = int(n * 0.95)

	train_tokens = torch.tensor(all_tokens[:train_end], dtype=torch.long)
	val_tokens = torch.tensor(all_tokens[train_end:val_end], dtype=torch.long)
	test_tokens = torch.tensor(all_tokens[val_end:], dtype=torch.long)

	print(f"3. Saving splits...")
	print(f"   Train: {len(train_tokens):,} tokens")
	print(f"   Val:   {len(val_tokens):,} tokens")
	print(f"   Test:  {len(test_tokens):,} tokens")

	torch.save(train_tokens, os.path.join(save_dir, "train_tokens.pt"))
	torch.save(val_tokens, os.path.join(save_dir, "validation_tokens.pt"))
	torch.save(test_tokens, os.path.join(save_dir, "test_tokens.pt"))

	tokenizer.save_pretrained(save_dir)
	torch.save({
		"seq_len": seq_len,
		"vocab_size": tokenizer.vocab_size,
		"target_tokens": target_tokens,
		"actual_tokens": n,
		"seed": seed
	}, os.path.join(save_dir, "metadata.pt"))

	print("Done!")


def get_dataloaders(task="classification",
                    batch_size=64,
                    seq_len=256,
                    classification_dataset="ag_news",
                    lm_dataset="fineweb",
                    is_ddp=False):

	if task == "classification":
		data_dir = f"./data/processed_{classification_dataset}"
		train, val, test, vocab, meta = get_classification_dataloaders(
			dataset_name=classification_dataset,
			data_dir=data_dir,
			batch_size=batch_size,
			val_split=0.1,
			is_ddp=is_ddp
		)
		return train, val, test, vocab, {
			"task": "classification",
			"dataset": classification_dataset,
			"num_classes": meta["num_classes"],
		}

	elif task == "language_modelling":
		if lm_dataset == "fineweb":
			data_dir = "./data/processed_fineweb"
		elif lm_dataset == "wikitext":
			data_dir = "./data/processed_wikitext2"
		else:
			raise ValueError(f"Unknown lm_dataset: {lm_dataset}")

		train, val, test, vocab = get_lm_dataloaders(
			data_dir=data_dir,
			batch_size=batch_size,
			seq_len=seq_len,
			is_ddp=is_ddp
		)
		return train, val, test, vocab, {
			"task": "language_modelling",
			"seq_len": seq_len,
			"dataset": lm_dataset
		}

	else:
		raise ValueError(f"Unknown task: {task}")

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--task",
	                    choices=["ag_news", "wikitext", "fineweb"],
	                    default="ag_news",
	                    help="Dataset to prepare")
	parser.add_argument("--tokens", type=int, default=30_000_000,
	                    help="Target tokens for FineWeb (default: 30M)")
	parser.add_argument("--seq_len", type=int, default=256,
	                    help="Sequence length (default: 256)")
	parser.add_argument("--seed", type=int, default=42,
	                    help="Random seed for FineWeb sampling")
	args = parser.parse_args()

	if args.task == "ag_news":
		prepare_classification_dataset("ag_news", max_length=512)
		train, val, test, vocab, meta = get_classification_dataloaders("ag_news")
		batch = next(iter(train))
		print(f"Input Shape: {batch['input_ids'].shape}")
		print(f"Vocab Size: {vocab}")
		print(f"Num Classes: {meta['num_classes']}")

	elif args.task == "wikitext":
		prepare_wikitext(seq_len=args.seq_len)
		tr, val, te, vocab = get_lm_dataloaders("./data/processed_wikitext2", seq_len=args.seq_len)
		batch = next(iter(tr))
		print(f"Input Shape: {batch['input_ids'].shape}")
		print(f"Target Shape: {batch['targets'].shape}")
		print(f"Vocab Size: {vocab}")

	elif args.task == "fineweb":
		prepare_fineweb(
			target_tokens=args.tokens,
			seq_len=args.seq_len,
			seed=args.seed
		)
		tr, val, te, vocab = get_lm_dataloaders("./data/processed_fineweb", seq_len=args.seq_len)
		batch = next(iter(tr))
		print(f"Input Shape: {batch['input_ids'].shape}")
		print(f"Target Shape: {batch['targets'].shape}")
		print(f"Vocab Size: {vocab}")