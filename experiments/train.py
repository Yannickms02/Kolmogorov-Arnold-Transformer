import time
import math
import torch
import torch.optim as optim
import wandb
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


class ClassificationTrainer:
	"""Classification Trainer (AG_NEWS)"""
	def __init__(self, model, train_loader, val_loader, learning_rate, device, config):
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		self.config = config

		self.optimizer = optim.AdamW(
			self.model.parameters(),
			lr=learning_rate,
			weight_decay=config.get("weight_decay", 0.01)
		)
		self.criterion = torch.nn.CrossEntropyLoss()

		self.global_step = 0

		# Calculate steps for scheduler
		accumulation_steps = config["batch_size"] // config["mini_batch_size"]
		steps_per_epoch = len(train_loader) // accumulation_steps
		total_steps = steps_per_epoch * config["max_epochs"]

		self.scheduler = get_cosine_schedule_with_warmup(
			self.optimizer,
			num_warmup_steps=config["warmup_steps"],
			num_training_steps=total_steps
		)
		self.scaler = torch.amp.GradScaler(self.device.type)

	def train_epoch(self, epoch_idx, mini_batch_size=16, target_batch_size=64):
		self.model.train()
		total_loss = 0
		correct = 0
		total = 0
		start_time = time.time()

		accumulation_steps = target_batch_size // mini_batch_size
		self.optimizer.zero_grad()

		# Only show progress bar on master process
		disable_tqdm = (wandb.run is None)
		progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx + 1} [Train]", disable=disable_tqdm)

		log_interval = 10

		for i, batch in enumerate(progress_bar):
			input_ids = batch['input_ids'].to(self.device)
			labels = batch['labels'].to(self.device)

			if self.device.type == 'cuda':
				with torch.amp.autocast(self.device.type):
					outputs = self.model(input_ids)
					loss = self.criterion(outputs, labels)
			else:
				outputs = self.model(input_ids)
				loss = self.criterion(outputs, labels)

			loss = loss / accumulation_steps

			if self.device.type == 'cuda':
				self.scaler.scale(loss).backward()
			else:
				loss.backward()

			if (i + 1) % accumulation_steps == 0:
				if self.device.type == 'cuda':
					self.scaler.unscale_(self.optimizer)
					grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
					self.scaler.step(self.optimizer)
					self.scaler.update()
				else:
					grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
					self.optimizer.step()

				self.scheduler.step()
				self.optimizer.zero_grad()
				self.global_step += 1

				# Log only if wandb is initialized (Rank 0)
				if self.global_step % log_interval == 0 and wandb.run is not None:
					wandb.log({
						"train/loss_step": loss.item() * accumulation_steps,
						"train/grad_norm": grad_norm,
						"train/lr": self.scheduler.get_last_lr()[0],
						"train/epoch_float": epoch_idx + (i / len(self.train_loader))
					}, step=self.global_step)

			current_loss = loss.item() * accumulation_steps
			total_loss += current_loss
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

			if not disable_tqdm:
				progress_bar.set_postfix({
					'loss': f'{current_loss:.4f}',
					'acc': f'{100. * correct / total:.2f}%',
					'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
				})

		return total_loss / len(self.train_loader), 100. * correct / total, time.time() - start_time

	@torch.no_grad()
	def evaluate(self, loader=None):
		if loader is None:
			loader = self.val_loader

		self.model.eval()
		total_loss = 0
		correct = 0
		total = 0
		all_preds = []
		all_labels = []

		for batch in loader:
			input_ids = batch['input_ids'].to(self.device)
			labels = batch['labels'].to(self.device)

			outputs = self.model(input_ids)
			loss = self.criterion(outputs, labels)

			total_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

			all_preds.extend(predicted.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

		acc = 100. * correct / total
		f1 = 100. * f1_score(all_labels, all_preds, average='macro')

		return total_loss / len(loader), acc, f1


class LanguageModelTrainer:
	"""Wikitext/Fineweb Trainer (Language Model)"""
	def __init__(self, model, train_loader, val_loader, learning_rate, device, config):
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		self.config = config

		self.optimizer = optim.AdamW(
			self.model.parameters(),
			lr=learning_rate,
			weight_decay=config.get("weight_decay", 0.01)
		)

		self.global_step = 0

		# Calculate steps
		accumulation_steps = config["batch_size"] // config["mini_batch_size"]
		steps_per_epoch = len(train_loader) // accumulation_steps
		total_steps = steps_per_epoch * config["max_epochs"]

		self.scheduler = get_cosine_schedule_with_warmup(
			self.optimizer,
			num_warmup_steps=config["warmup_steps"],
			num_training_steps=total_steps
		)
		self.scaler = torch.amp.GradScaler(self.device.type)

	def train_epoch(self, epoch_idx, mini_batch_size=16, target_batch_size=64):
		self.model.train()
		total_loss = 0
		total_tokens = 0
		start_time = time.time()

		accumulation_steps = target_batch_size // mini_batch_size
		self.optimizer.zero_grad()

		# Hide progress bar on non-master ranks
		disable_tqdm = (wandb.run is None)
		progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx + 1} [Train]", disable=disable_tqdm)
		log_interval = 10

		running_loss = 0
		running_tokens = 0

		for i, batch in enumerate(progress_bar):
			input_ids = batch['input_ids'].to(self.device)
			targets = batch['targets'].to(self.device)

			if self.device.type == 'cuda':
				with torch.amp.autocast(self.device.type):
					_, loss = self.model(input_ids, targets)
			else:
				_, loss = self.model(input_ids, targets)

			loss = loss / accumulation_steps

			if self.device.type == 'cuda':
				self.scaler.scale(loss).backward()
			else:
				loss.backward()

			if (i + 1) % accumulation_steps == 0:
				if self.device.type == 'cuda':
					self.scaler.unscale_(self.optimizer)
					grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
					self.scaler.step(self.optimizer)
					self.scaler.update()
				else:
					grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
					self.optimizer.step()

				self.scheduler.step()
				self.optimizer.zero_grad()
				self.global_step += 1

				# Log only if wandb is initialized (Rank 0)
				if self.global_step % log_interval == 0 and wandb.run is not None:
					current_step_loss = loss.item() * accumulation_steps
					wandb.log({
						"train/loss_step": current_step_loss,
						"train/ppl_step": math.exp(min(current_step_loss, 20)),
						"train/grad_norm": grad_norm,
						"train/lr": self.scheduler.get_last_lr()[0],
						"train/epoch_float": epoch_idx + (i / len(self.train_loader))
					}, step=self.global_step)

			# Metrics
			current_loss = loss.item() * accumulation_steps
			num_tokens = targets.numel()

			total_loss += current_loss * num_tokens
			total_tokens += num_tokens

			running_loss += current_loss * num_tokens
			running_tokens += num_tokens
			if running_tokens > 50000:
				running_loss = current_loss * num_tokens
				running_tokens = num_tokens

			if not disable_tqdm:
				running_ppl = math.exp(min(running_loss / running_tokens, 20))
				progress_bar.set_postfix({
					'loss': f'{running_loss / running_tokens:.3f}',
					'ppl': f'{running_ppl:.1f}',
					'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
				})

		avg_loss = total_loss / total_tokens
		perplexity = math.exp(min(avg_loss, 20))

		return avg_loss, perplexity, time.time() - start_time

	@torch.no_grad()
	def evaluate(self, loader=None):
		if loader is None:
			loader = self.val_loader

		self.model.eval()
		total_loss = 0
		total_tokens = 0

		for batch in loader:
			input_ids = batch['input_ids'].to(self.device)
			targets = batch['targets'].to(self.device)

			_, loss = self.model(input_ids, targets)

			num_tokens = targets.numel()
			total_loss += loss.item() * num_tokens
			total_tokens += num_tokens

		avg_loss = total_loss / total_tokens
		perplexity = math.exp(min(avg_loss, 20))

		return avg_loss, perplexity


def get_trainer(task, model, train_loader, val_loader, lr, device, config):
	if task == "classification":
		return ClassificationTrainer(model, train_loader, val_loader, lr, device, config)
	elif task == "language_modelling":
		return LanguageModelTrainer(model, train_loader, val_loader, lr, device, config)
	else:
		raise ValueError(f"Unknown task: {task}")