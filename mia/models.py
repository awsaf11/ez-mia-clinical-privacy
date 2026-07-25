from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from .ez_score import error_zone_pos_neg_sum_ratio
from .min_k_score import min_k_percent_score


def _progress(sequence, **kwargs):
	disable = kwargs.pop("disable", None)
	if disable is None:
		disable = not (sys.stderr.isatty() or sys.stdout.isatty())
	if "total" not in kwargs and hasattr(sequence, "__len__"):
		try:
			kwargs["total"] = len(sequence)
		except TypeError:
			pass
	kwargs.setdefault("dynamic_ncols", True)
	kwargs.setdefault("leave", True)
	return tqdm(sequence, disable=disable, **kwargs)


def build_tokenizer(model_name: str = "gpt2"):
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	if tokenizer.pad_token is None:
		tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
	return tokenizer


def _fallback_target_modules(model_name: str) -> List[str]:
	name = model_name.lower()
	llama_like = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
	gpt2_like = ["c_attn", "c_proj", "c_fc"]
	gptj_like = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "proj", "c_attn", "c_proj"]
	if "llama" in name or "codellama" in name:
		return llama_like
	if "gpt-j" in name or "gptj" in name:
		return gptj_like
	if "gpt2" in name:
		return gpt2_like
	return list(set(llama_like + gptj_like + gpt2_like))


def _patch_torch_load_safety() -> bool:
	try:
		from transformers import utils as hf_utils
	except Exception:
		return False
	checker = getattr(hf_utils, "check_torch_load_is_safe", None)
	if checker is None:
		return False
	if getattr(hf_utils, "_mia_patched_check_torch_load_is_safe", False):
		return True

	def _noop_check_torch_load_is_safe():
		return None

	hf_utils.check_torch_load_is_safe = _noop_check_torch_load_is_safe
	hf_utils._mia_patched_check_torch_load_is_safe = True
	return True


def _load_auto_model(model_name: str, dtype=None, **kwargs):
	load_kwargs = {"low_cpu_mem_usage": True}
	if dtype is not None:
		load_kwargs["torch_dtype"] = dtype
	load_kwargs.update(kwargs)
	try:
		return AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, **load_kwargs)
	except Exception as e:
		msg = str(e).lower()

		def _load_bin_with_patch():
			patched = _patch_torch_load_safety()
			if patched:
				tqdm.write("[load] torch<2.6 detected; bypassing torch.load safety check to load weights.")
			else:
				raise RuntimeError(
					"Unable to bypass torch>=2.6 requirement. Please upgrade torch to >=2.6 or use safetensors checkpoints."
				) from e
			return AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=False, **load_kwargs)

		if "safetensor" in msg or "safetensors" in msg or "torch to at least v2.6" in msg or "torch>=2.6" in msg:
			return _load_bin_with_patch()
		raise


def _maybe_apply_lora(model, target_modules: list[str], r: int, alpha: int, dropout: float):
	if not target_modules:
		raise ValueError("lora_target_modules must be specified when using LoRA")
	lcfg = LoraConfig(
		r=r,
		lora_alpha=alpha,
		lora_dropout=dropout,
		target_modules=target_modules,
		bias="none",
		task_type="CAUSAL_LM",
	)
	return get_peft_model(model, lcfg)


def build_model(
	model_name: str,
	tokenizer,
	device: torch.device,
	*,
	use_lora: bool = False,
	lora_r: int = 8,
	lora_alpha: int = 16,
	lora_dropout: float = 0.05,
	lora_target_modules: list[str] | None = None,
) -> AutoModelForCausalLM:
	model = _load_auto_model(model_name)
	if tokenizer.pad_token is not None and getattr(model.config, "pad_token_id", None) is None:
		model.resize_token_embeddings(len(tokenizer))
		model.config.pad_token_id = tokenizer.pad_token_id

	if hasattr(model, "gradient_checkpointing_disable"):
		try:
			model.gradient_checkpointing_disable()
		except Exception:
			pass

	if use_lora:
		model = _maybe_apply_lora(model, lora_target_modules, lora_r, lora_alpha, lora_dropout)

	model.to(device)
	return model


def prepare_lm_dataloader(tokenizer, texts: List[str], batch_size: int, device: torch.device, *, sequence_length: int):
	class _TextDataset(Dataset):
		def __init__(self, texts_list):
			self.texts = texts_list

		def __len__(self):
			return len(self.texts)

		def __getitem__(self, idx):
			return self.texts[idx]

	def _collate_text(batch: List[str]):
		encodings = tokenizer(
			batch,
			padding="max_length",
			truncation=True,
			max_length=sequence_length,
			return_tensors="pt",
		)

		input_ids = encodings["input_ids"]
		attention_mask = encodings["attention_mask"]
		labels = input_ids.clone()
		labels[attention_mask == 0] = -100

		return {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"labels": labels,
		}

	dataset = _TextDataset(texts)
	pin_memory = device.type == "cuda"

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		collate_fn=_collate_text,
		pin_memory=pin_memory,
	)

def finetune_target(
	model,
	train_dataloader,
	epochs: int,
	lr: float,
	warmup_steps: int = 0,
	*,
	val_dataloader: DataLoader | None = None,
	load_best_on_val: bool = True,
	epoch_callback=None,
) -> float:
	if load_best_on_val and val_dataloader is None:
		raise ValueError("Validation dataloader is required when load_best_on_val=True.")

	device = model.device
	model.train()

	if hasattr(model.config, "use_cache"):
		model.config.use_cache = False

	optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
	total_steps = epochs * max(len(train_dataloader), 1)
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=warmup_steps,
		num_training_steps=total_steps,
	)

	last_train_loss = 0.0
	best_val_loss: float | None = None
	best_state_dir: Path | None = None
	best_state_path: Path | None = None

	if load_best_on_val and val_dataloader is not None:
		best_state_dir = Path(tempfile.mkdtemp(prefix="finetune_best_"))
		best_state_path = best_state_dir / "best.pt"

	try:
		for epoch_idx in range(epochs):
			total = 0.0
			n = 0
			epoch_bar = _progress(
				train_dataloader,
				desc=f"Train epoch {epoch_idx + 1}/{epochs}",
				unit="batch",
			)

			for batch in epoch_bar:
				batch = {
					k: (v.to(device) if isinstance(v, torch.Tensor) else v)
					for k, v in batch.items()
				}
				outputs = model(**batch)
				loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

				optimizer.zero_grad(set_to_none=True)
				loss.backward()
				optimizer.step()
				scheduler.step()

				total += float(loss.detach().cpu().item())
				n += 1
				epoch_bar.set_postfix(loss=f"{(total / max(n, 1)):.4f}")

			epoch_bar.close()
			last_train_loss = total / max(n, 1)

			val_avg = None
			if val_dataloader is not None:
				model.eval()
				with torch.no_grad():
					vtotal = 0.0
					vn = 0
					val_bar = _progress(
						val_dataloader,
						desc=f"Val epoch {epoch_idx + 1}/{epochs}",
						unit="batch",
					)

					for vbatch in val_bar:
						vbatch = {
							k: (v.to(device) if isinstance(v, torch.Tensor) else v)
							for k, v in vbatch.items()
						}
						vout = model(**vbatch)
						vloss = vout.loss if hasattr(vout, "loss") else vout[0]
						vtotal += float(vloss.detach().cpu().item())
						vn += 1
						val_bar.set_postfix(loss=f"{(vtotal / max(vn, 1)):.4f}")

					val_bar.close()
					val_avg = vtotal / max(vn, 1)

				model.train()

				if best_val_loss is None or val_avg < best_val_loss:
					best_val_loss = val_avg
					if load_best_on_val and best_state_path is not None:
						torch.save(model.state_dict(), best_state_path)

			if val_avg is None:
				tqdm.write(
					f"[train] epoch {epoch_idx + 1}/{epochs} - loss: {last_train_loss:.6f}"
				)
			else:
				tqdm.write(
					f"[train] epoch {epoch_idx + 1}/{epochs} - loss: {last_train_loss:.6f} - val_loss: {val_avg:.6f}"
				)

			if epoch_callback is not None:
				epoch_callback(
					epoch=epoch_idx + 1,
					model=model,
					train_loss=last_train_loss,
					val_loss=val_avg,
				)

		selected_loss = best_val_loss if best_val_loss is not None else last_train_loss

		if load_best_on_val:
			if best_state_path is None or best_val_loss is None:
				raise RuntimeError("Validation loss was not computed; cannot load best checkpoint.")
			state = torch.load(best_state_path, map_location=device)
			model.load_state_dict(state)
			selected_loss = best_val_loss
			tqdm.write(f"[train] loaded best validation checkpoint (val_loss={best_val_loss:.6f})")

		return float(selected_loss)

	finally:
		if best_state_dir is not None:
			shutil.rmtree(best_state_dir, ignore_errors=True)


def build_sft_reference(
	tokenizer,
	device: torch.device,
	*,
	base_model_name: str = "gpt2",
	texts: list[str],
	epochs: int = 1,
	batch_size: int = 16,
	lr: float = 1e-4,
	val_texts: list[str] | None = None,
	sequence_length: int = 128,
	use_lora: bool = False,
	lora_r: int = 8,
	lora_alpha: int = 16,
	lora_dropout: float = 0.05,
	lora_target_modules: list[str] | None = None,
):
	ref_model = build_model(
		base_model_name,
		tokenizer,
		device,
		use_lora=use_lora,
		lora_r=lora_r,
		lora_alpha=lora_alpha,
		lora_dropout=lora_dropout,
		lora_target_modules=lora_target_modules,
	)

	if val_texts and len(val_texts) > 0:
		train_dataloader = prepare_lm_dataloader(tokenizer, texts, batch_size=batch_size, device=device, sequence_length=sequence_length)
		validation_dataloader = prepare_lm_dataloader(tokenizer, val_texts, batch_size=batch_size, device=device, sequence_length=sequence_length)
		_ = finetune_target(ref_model, train_dataloader, epochs=epochs, lr=lr, val_dataloader=validation_dataloader)
	else:
		dataloader = prepare_lm_dataloader(tokenizer, texts, batch_size=batch_size, device=device, sequence_length=sequence_length)
		_ = finetune_target(ref_model, dataloader, epochs=epochs, lr=lr)

	return ref_model


def _extract_stats_for_model(
	model,
	tokenizer,
	texts: List[str],
	device: torch.device,
	sequence_length: int,
	batch_size: int,
	defense: str = "none",
	noise_std: float = 0.0,
	risk_k_percent: float = 20.0,
	smoothing_alpha: float = 0.8,
	adaptive_beta: float = 2.0,
) -> List[Dict[str, Any]]:
	model.eval()

	if hasattr(model.config, "use_cache"):
		model.config.use_cache = True

	model.to(device)

	stats_out = []
	with torch.no_grad():
		for i in range(0, len(texts), batch_size):
			batch_texts = texts[i:i + batch_size]
			encodings = tokenizer(
				batch_texts,
				padding="max_length",
				truncation=True,
				max_length=sequence_length,
				return_tensors="pt",
			)
			input_ids = encodings["input_ids"].to(device)
			attention_mask = encodings["attention_mask"].to(device)

			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
			logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

			if defense == "output_perturbation" and noise_std > 0:
				logits = logits + torch.randn_like(logits) * noise_std

			pred_logits = logits[:, :-1, :]
			target_ids = input_ids[:, 1:]
			mask = attention_mask[:, 1:]

			if defense == "bottom_k_smoothing":
				lp_initial = F.log_softmax(pred_logits, dim=-1)
				correct_initial = lp_initial.gather(
					-1,
					target_ids.unsqueeze(-1),
				).squeeze(-1)

				valid_correct = correct_initial.masked_fill(mask == 0, float("inf"))
				num_tokens = mask.sum(dim=1)

				k_counts = torch.clamp(
					(num_tokens.float() * (risk_k_percent / 100.0)).ceil().long(),
					min=1,
				)

				for b in range(pred_logits.shape[0]):
					k = int(k_counts[b].item())
					valid_positions = torch.where(mask[b] > 0)[0]

					if len(valid_positions) == 0:
						continue

					k = min(k, len(valid_positions))

					bottom_positions = torch.topk(
						valid_correct[b, valid_positions],
						k=k,
						largest=False,
					).indices

					selected_positions = valid_positions[bottom_positions]
					selected_logits = pred_logits[b, selected_positions, :]

					selected_correct = correct_initial[b, selected_positions]
					seq_valid_correct = correct_initial[b, valid_positions]

					mu = seq_valid_correct.mean()
					sigma = seq_valid_correct.std().clamp(min=1e-6)

					risk_scores = (mu - selected_correct) / sigma
					adaptive_alpha = torch.sigmoid(adaptive_beta * risk_scores)

					adaptive_alpha = smoothing_alpha * adaptive_alpha
					adaptive_alpha = adaptive_alpha.view(-1, 1)

					mean_logits = selected_logits.mean(
						dim=-1,
						keepdim=True,
					)

					pred_logits[b, selected_positions, :] = (
						(1 - adaptive_alpha) * selected_logits
						+ adaptive_alpha * mean_logits
					)

			lp = F.log_softmax(pred_logits, dim=-1)
			correct = lp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

			_, top1_idx = torch.topk(lp, k=1, dim=-1)
			top1_idx = top1_idx.squeeze(-1)

			batch_count = input_ids.shape[0]
			for b in range(batch_count):
				stats_out.append({
					"correct": correct[b].cpu(),
					"top1_idx": top1_idx[b].cpu(),
					"target_ids": target_ids[b].cpu(),
					"mask": mask[b].cpu(),
				})

	return stats_out

def compute_utility_metrics(
	model,
	tokenizer,
	texts: List[str],
	device: torch.device,
	sequence_length: int = 128,
	batch_size: int = 32,
	defense: str = "none",
	noise_std: float = 0.0,
	risk_k_percent: float = 20.0,
	smoothing_alpha: float = 0.8,
	adaptive_beta: float = 2.0,
) -> Dict[str, float]:
	"""
	Compare the original target-model outputs against defended outputs.

	Metrics:
	- Baseline perplexity
	- Defended perplexity
	- Relative perplexity change
	- Top-1 prediction agreement
	- Mean Jensen-Shannon divergence
	"""

	if not texts:
		raise ValueError("Utility evaluation requires at least one text.")

	model.eval()
	model.to(device)

	if hasattr(model.config, "use_cache"):
		model.config.use_cache = True

	# JS divergence requires several vocabulary-sized tensors.
	# Use a smaller batch to reduce GPU-memory pressure.
	utility_batch_size = max(1, min(int(batch_size), 4))

	baseline_nll_sum = 0.0
	defended_nll_sum = 0.0
	agreement_count = 0
	valid_token_count = 0
	js_sum = 0.0

	with torch.no_grad():
		for start in range(0, len(texts), utility_batch_size):
			batch_texts = texts[start:start + utility_batch_size]

			encodings = tokenizer(
				batch_texts,
				padding="max_length",
				truncation=True,
				max_length=sequence_length,
				return_tensors="pt",
			)

			input_ids = encodings["input_ids"].to(device)
			attention_mask = encodings["attention_mask"].to(device)

			outputs = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
			)

			logits = (
				outputs.logits
				if hasattr(outputs, "logits")
				else outputs[0]
			)

			# Original target-model logits.
			baseline_pred_logits = logits[:, :-1, :]
			target_ids = input_ids[:, 1:]
			valid_mask = attention_mask[:, 1:].bool()

			# The defense must operate on a copy so the baseline remains intact.
			defended_pred_logits = baseline_pred_logits.clone()

			if defense == "output_perturbation" and noise_std > 0:
				defended_pred_logits = (
					defended_pred_logits
					+ torch.randn_like(defended_pred_logits) * noise_std
				)

			elif defense == "bottom_k_smoothing":
				initial_log_probs = F.log_softmax(
					defended_pred_logits,
					dim=-1,
				)

				correct_initial = initial_log_probs.gather(
					-1,
					target_ids.unsqueeze(-1),
				).squeeze(-1)

				valid_correct = correct_initial.masked_fill(
					~valid_mask,
					float("inf"),
				)

				num_tokens = valid_mask.sum(dim=1)

				k_counts = torch.clamp(
					(
						num_tokens.float()
						* (risk_k_percent / 100.0)
					).ceil().long(),
					min=1,
				)

				for b in range(defended_pred_logits.shape[0]):
					valid_positions = torch.where(valid_mask[b])[0]

					if valid_positions.numel() == 0:
						continue

					k = min(
						int(k_counts[b].item()),
						int(valid_positions.numel()),
					)

					bottom_indices = torch.topk(
						valid_correct[b, valid_positions],
						k=k,
						largest=False,
					).indices

					selected_positions = valid_positions[bottom_indices]

					selected_logits = defended_pred_logits[
						b,
						selected_positions,
						:,
					]

					selected_correct = correct_initial[
						b,
						selected_positions,
					]

					sequence_correct = correct_initial[
						b,
						valid_positions,
					]

					mu = sequence_correct.mean()
					sigma = sequence_correct.std().clamp(min=1e-6)

					risk_scores = (
						mu - selected_correct
					) / sigma

					adaptive_alpha = smoothing_alpha * torch.sigmoid(
						adaptive_beta * risk_scores
					)

					adaptive_alpha = adaptive_alpha.view(-1, 1)

					mean_logits = selected_logits.mean(
						dim=-1,
						keepdim=True,
					)

					defended_pred_logits[
						b,
						selected_positions,
						:
					] = (
						(1.0 - adaptive_alpha) * selected_logits
						+ adaptive_alpha * mean_logits
					)

			# Log-probability distributions.
			baseline_log_probs = F.log_softmax(
				baseline_pred_logits,
				dim=-1,
			)

			defended_log_probs = F.log_softmax(
				defended_pred_logits,
				dim=-1,
			)

			# Correct-token log probabilities for perplexity.
			baseline_correct = baseline_log_probs.gather(
				-1,
				target_ids.unsqueeze(-1),
			).squeeze(-1)

			defended_correct = defended_log_probs.gather(
				-1,
				target_ids.unsqueeze(-1),
			).squeeze(-1)

			baseline_nll_sum += float(
				(-baseline_correct[valid_mask]).sum().item()
			)

			defended_nll_sum += float(
				(-defended_correct[valid_mask]).sum().item()
			)

			# Top-1 agreement with the undefended target model.
			baseline_top1 = baseline_pred_logits.argmax(dim=-1)
			defended_top1 = defended_pred_logits.argmax(dim=-1)

			agreement_count += int(
				(
					baseline_top1[valid_mask]
					== defended_top1[valid_mask]
				).sum().item()
			)

			batch_valid_tokens = int(valid_mask.sum().item())
			valid_token_count += batch_valid_tokens

			# Jensen-Shannon divergence in log space.
			log_mixture = torch.logaddexp(
				baseline_log_probs,
				defended_log_probs,
			) - torch.log(
				torch.tensor(
					2.0,
					device=device,
					dtype=baseline_log_probs.dtype,
				)
			)

			kl_baseline_to_mixture = torch.sum(
				baseline_log_probs.exp()
				* (baseline_log_probs - log_mixture),
				dim=-1,
			)

			kl_defended_to_mixture = torch.sum(
				defended_log_probs.exp()
				* (defended_log_probs - log_mixture),
				dim=-1,
			)

			js_per_token = 0.5 * (
				kl_baseline_to_mixture
				+ kl_defended_to_mixture
			)

			js_sum += float(js_per_token[valid_mask].sum().item())

			# Release large vocabulary tensors before the next batch.
			del (
				outputs,
				logits,
				baseline_pred_logits,
				defended_pred_logits,
				baseline_log_probs,
				defended_log_probs,
				log_mixture,
			)

			if device.type == "cuda":
				torch.cuda.empty_cache()

	if valid_token_count == 0:
		raise ValueError(
			"Utility evaluation found no valid next-token positions."
		)

	mean_baseline_nll = baseline_nll_sum / valid_token_count
	mean_defended_nll = defended_nll_sum / valid_token_count

	baseline_perplexity = float(
		torch.exp(torch.tensor(mean_baseline_nll)).item()
	)

	defended_perplexity = float(
		torch.exp(torch.tensor(mean_defended_nll)).item()
	)

	if baseline_perplexity > 0:
		perplexity_change_percent = (
			(defended_perplexity - baseline_perplexity)
			/ baseline_perplexity
		) * 100.0
	else:
		perplexity_change_percent = float("nan")

	top1_agreement = agreement_count / valid_token_count
	mean_js_divergence = js_sum / valid_token_count

	model.to("cpu")

	if device.type == "cuda":
		torch.cuda.empty_cache()

	return {
		"baseline_perplexity": baseline_perplexity,
		"defended_perplexity": defended_perplexity,
		"perplexity_change_percent": float(
			perplexity_change_percent
		),
		"top1_agreement": float(top1_agreement),
		"js_divergence": float(mean_js_divergence),
		"utility_token_count": int(valid_token_count),
		"utility_example_count": int(len(texts)),
	}			




def compute_ez_scores(
	tokenizer,
	target_model,
	reference_model,
	texts: List[str],
	device: torch.device,
	sequence_length: int = 128,
	batch_size: int = 32,
	defense: str = "none",
	noise_std: float = 0.0,
	risk_k_percent: float = 20.0,
	smoothing_alpha: float = 0.8,
	adaptive_beta: float = 2.0,
) -> List[float]:
	t_stats_list = _extract_stats_for_model(
		target_model,
		tokenizer,
		texts,
		device,
		sequence_length,
		batch_size,
		defense=defense,
		noise_std=noise_std,
		risk_k_percent=risk_k_percent,
		smoothing_alpha=smoothing_alpha,
		adaptive_beta=adaptive_beta,
	)

	target_model.to("cpu")
	torch.cuda.empty_cache()

	r_stats_list = _extract_stats_for_model(
		reference_model,
		tokenizer,
		texts,
		device,
		sequence_length,
		batch_size,
		defense="none",
	    noise_std=0.0,
	)

	scores: List[float] = []

	for t_s, r_s in zip(t_stats_list, r_stats_list):
		ez = error_zone_pos_neg_sum_ratio(
			t_s["correct"],
			r_s["correct"],
			t_s["top1_idx"],
			t_s["target_ids"],
			t_s["mask"],
			ignore_bos=True,
			min_tokens=2,
		)
		scores.append(float(ez))

	return scores


def compute_min_k_scores(
	tokenizer,
	target_model,
	texts: List[str],
	device: torch.device,
	sequence_length: int = 128,
	batch_size: int = 32,
	k_percent: float = 20.0,
	defense: str = "none",
	noise_std: float = 0.0,
	risk_k_percent: float = 20.0,
	smoothing_alpha: float = 0.8,
	adaptive_beta: float = 2.0,
) -> List[float]:
	t_stats_list = _extract_stats_for_model(
		target_model,
		tokenizer,
		texts,
		device,
		sequence_length,
		batch_size,
		defense=defense,
		noise_std=noise_std,
		risk_k_percent=risk_k_percent,
		smoothing_alpha=smoothing_alpha,
		adaptive_beta=adaptive_beta,
	)

	target_model.to("cpu")
	torch.cuda.empty_cache()

	scores: List[float] = []

	for t_s in t_stats_list:
		score = min_k_percent_score(
			t_s["correct"],
			t_s["mask"],
			k_percent=k_percent,
			ignore_bos=True,
			min_tokens=2,
		)
		scores.append(float(score))

	return scores

def build_distillation_reference(
	tokenizer,
	target_model,
	seed_texts: List[str],
	device: torch.device,
	*,
	max_prompts: int = 200,
	completions: int = 1,
	max_new_tokens: int = 64,
	temperature: float = 0.9,
	top_p: float = 0.9,
	input_max_tokens: int = 128,
	train_epochs: int = 1,
	train_batch: int = 8,
	train_lr: float = 5e-5,
	base_model_name: str = "gpt2",
	use_lora_for_ref: bool = False,
	lora_r: int = 8,
	lora_alpha: int = 16,
	lora_dropout: float = 0.05,
	lora_target_modules: list[str] | None = None,
	val_texts: list[str] | None = None,
	sequence_length: int = 128,
):
	"""Create a distillation-style reference model by generating synthetic texts with the target and fine-tuning a fresh base model on them."""
	def _truncate_to_tokens(text: str, max_tokens: int) -> str:
		encodings = tokenizer(text, truncation=True, max_length=max_tokens, return_tensors="pt")
		return tokenizer.decode(encodings["input_ids"][0], skip_special_tokens=True)

	prompts = [_truncate_to_tokens(t, input_max_tokens) for t in seed_texts[:max_prompts]]

	target_model.to(device)
	target_model.eval()

	if hasattr(target_model.config, "use_cache"):
		target_model.config.use_cache = True

	synthetic_texts: List[str] = []
	with torch.no_grad():
		prompt_bar = _progress(prompts, desc="Distill prompts", unit="prompt")
		for p in prompt_bar:
			if not p.strip():
				continue
			inputs = tokenizer(p, return_tensors="pt").to(device)
			generated = target_model.generate(
				**inputs,
				do_sample=True,
				temperature=float(temperature),
				top_p=float(top_p),
				max_new_tokens=int(max_new_tokens),
				num_return_sequences=int(completions),
				pad_token_id=tokenizer.pad_token_id,
				eos_token_id=tokenizer.eos_token_id,
			)
			for sequence in generated:
				text = tokenizer.decode(sequence, skip_special_tokens=True)
				if len(text) > 0:
					synthetic_texts.append(text)

	synthetic_texts = list(dict.fromkeys(synthetic_texts))
	tqdm.write(f"[distil] Generated {len(synthetic_texts)} unique synthetic texts from {len(prompts)} prompts")

	target_model.to("cpu")
	torch.cuda.empty_cache()

	ref_model = build_model(
		base_model_name,
		tokenizer,
		device,
		use_lora=use_lora_for_ref,
		lora_r=lora_r,
		lora_alpha=lora_alpha,
		lora_dropout=lora_dropout,
		lora_target_modules=lora_target_modules,
	)

	class _SyntheticDataset(Dataset):
		def __init__(self, texts: List[str]):
			self.texts = texts
		def __len__(self):
			return len(self.texts)
		def __getitem__(self, idx):
			return self.texts[idx]
	def _collate_synthetic(batch: List[str]):
		encodings = tokenizer(batch, padding="max_length", truncation=True, max_length=sequence_length, return_tensors="pt")
		input_ids = encodings["input_ids"]
		attention_mask = encodings["attention_mask"]
		labels = input_ids.clone()
		labels[attention_mask == 0] = -100
		return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

	if len(synthetic_texts) > 0:
		if val_texts and len(val_texts) > 0:
			train_dataset = _SyntheticDataset(synthetic_texts)
			train_dataloader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, collate_fn=_collate_synthetic, pin_memory=(device.type=="cuda"))
			val_dataset = _SyntheticDataset(val_texts)
			val_dataloader = DataLoader(val_dataset, batch_size=train_batch, shuffle=False, collate_fn=_collate_synthetic, pin_memory=(device.type=="cuda"))
			_ = finetune_target(ref_model, train_dataloader, epochs=train_epochs, lr=train_lr, val_dataloader=val_dataloader)
		else:
			full_dataset = _SyntheticDataset(synthetic_texts)
			full_dataloader = DataLoader(full_dataset, batch_size=train_batch, shuffle=True, collate_fn=_collate_synthetic, pin_memory=(device.type=="cuda"))
			_ = finetune_target(ref_model, full_dataloader, epochs=train_epochs, lr=train_lr)
	return ref_model