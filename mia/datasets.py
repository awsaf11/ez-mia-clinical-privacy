from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from datasets import load_dataset as hf_load_dataset


@dataclass
class Example:
	id: str
	text: str
	label: int


_PREFIX_SOURCES = {
	"wikitext": {
		"type": "parquet",
		"data_files": ["https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-00000-of-00041.parquet"],
		"text_fields": ["text"],
		"streaming": False,
	},
	"ag_news": {
		"type": "hf_dataset",
		"name": "heegyu/news-category-dataset",
		"split": "train",
		"text_fields": ["short_description"],
		"streaming": True,
	},
	"xsum": {
		"type": "hf_dataset",
		"name": "abisee/cnn_dailymail",
		"config": "3.0.0",
		"split": "train",
		"text_fields": ["article"],
		"streaming": True,
	},
	"tokyotech-llm/swallow-code": {
		"type": "hf_dataset",
		"name": "codeparrot/codeparrot-clean",
		"split": "train",
		"text_fields": ["content"],
		"streaming": True,
	},
	"swallow-code": {  # alias
		"type": "hf_dataset",
		"name": "codeparrot/codeparrot-clean",
		"split": "train",
		"text_fields": ["content"],
		"streaming": True,
	},
	"mtsamples": {
		"type": "local_csv",
		"text_fields": ["transcription"],
	},
}

def _load_mtsamples_raw_texts() -> List[str]:
	data_path = Path("data/mtsamples.csv")
	if not data_path.exists():
		raise FileNotFoundError(
			"Expected MTSamples CSV at data/mtsamples.csv. "
			"Create a data/ folder and place mtsamples.csv there."
		)

	import pandas as pd

	df = pd.read_csv(data_path)

	if "transcription" not in df.columns:
		raise ValueError(
			f"Expected a 'transcription' column in MTSamples CSV. Found columns: {list(df.columns)}"
		)

	texts = (
		df["transcription"]
		.dropna()
		.astype(str)
		.map(lambda x: x.strip())
		.tolist()
	)

	texts = [t for t in texts if len(t.split()) >= 30]
	return texts

STREAM_SEQUENCE_BUFFER_TARGET = 200_000


def sample_mtsamples_partition(
	seed: int,
	*,
	target_eval_total: int,
	target_val_total: int,
	domain_train_total: int,
	domain_val_total: int,
	distil_max_prompts: int = 0,
	sequence_length: int = 128,
) -> dict[str, list[str]]:
	"""Create one deterministic, disjoint MTSamples partition for the full experiment.

	The returned subsets never overlap by exact sequence. This keeps target attack
	evaluation, target validation, reference-model training/validation, and
	distillation prompts separate while preserving the same clinical domain.
	"""
	raw_texts = _load_mtsamples_raw_texts()
	sequences = _texts_to_sequences_concat(
		raw_texts,
		sequence_length=sequence_length,
	)

	# MTSamples contains repeated/template-like clinical text. Remove exact
	# duplicate fixed-length sequences before partitioning so that no identical
	# sequence can appear in two experimental subsets.
	original_sequence_count = len(sequences)
	sequences = list(dict.fromkeys(sequences))
	duplicate_count = original_sequence_count - len(sequences)

	if not sequences:
		raise ValueError(
			"No usable MTSamples sequences were produced after preprocessing."
		)

	target_member_needed = max(0, int(target_eval_total) // 2)
	target_nonmember_needed = max(0, int(target_eval_total) // 2)
	target_val_needed = max(0, int(target_val_total))
	domain_member_needed = max(0, int(domain_train_total) // 2)
	domain_nonmember_needed = max(0, int(domain_train_total) // 2)
	domain_val_needed = max(0, int(domain_val_total))
	distil_needed = max(0, int(distil_max_prompts))

	counts = {
		"target_members": target_member_needed,
		"target_nonmembers": target_nonmember_needed,
		"target_validation": target_val_needed,
		"domain_members": domain_member_needed,
		"domain_nonmembers": domain_nonmember_needed,
		"domain_validation": domain_val_needed,
		"distillation_prompts": distil_needed,
	}
	total_needed = sum(counts.values())

	if len(sequences) < total_needed:
		raise ValueError(
			f"Requested {total_needed} disjoint MTSamples sequences of length "
			f"{sequence_length}, but only {len(sequences)} unique sequences "
			f"were available after removing {duplicate_count} duplicates. "
			"Reduce eval_total, train_total, val_total, distil_max_prompts, "
			"or sequence_length."
		)

	if len(sequences) != len(set(sequences)):
		raise RuntimeError("Internal error: MTSamples sequence deduplication failed.")

	rng = np.random.RandomState(seed + 17)
	rng.shuffle(sequences)

	partition: dict[str, list[str]] = {}
	cursor = 0
	for name, count in counts.items():
		partition[name] = sequences[cursor:cursor + count]
		cursor += count

	# Defensive exact-overlap assertion.
	seen: dict[str, str] = {}
	for name, subset in partition.items():
		for text in subset:
			previous = seen.get(text)
			if previous is not None:
				raise RuntimeError(
					f"MTSamples partition overlap detected between {previous} and {name}."
				)
			seen[text] = name

	return partition


_HF_DATASET_ALIASES = {
	"ag_news": "fancyzhx/ag_news",
	"xsum": "EdinburghNLP/xsum",
	"wikitext": "Salesforce/wikitext",
	"cnn_dailymail": "abisee/cnn_dailymail",
	# The old repository depends on a loading script rejected by recent
	# `datasets` releases. This replacement is standard-format Python code.
	"codeparrot/github-code-clean": "codeparrot/codeparrot-clean",
}


def load_dataset(*args, **kwargs):
	"""Load datasets through current canonical Hugging Face identifiers."""
	kwargs.pop("trust_remote_code", None)

	if args and isinstance(args[0], str):
		requested_name = args[0]
		canonical_name = _HF_DATASET_ALIASES.get(requested_name, requested_name)
		args = (canonical_name, *args[1:])

		# This option belonged to the old github-code-clean loading script.
		if canonical_name == "codeparrot/codeparrot-clean":
			kwargs.pop("languages", None)

	return hf_load_dataset(*args, **kwargs)



def _load_hf_split(
	dataset_name: str,
	*,
	config: str | None = None,
	split: str = "train",
	streaming: bool = False,
	**kwargs,
):
	"""Load one split without passing a positional None configuration."""
	if config is None:
		return load_dataset(
			dataset_name,
			split=split,
			streaming=streaming,
			**kwargs,
		)
	return load_dataset(
		dataset_name,
		config,
		split=split,
		streaming=streaming,
		**kwargs,
	)

def _collect_streaming_texts(
	text_iter,
	*,
	min_required: int,
	label: str,
	buffer_target: int | None = None,
) -> List[str]:
	"""Collect up to `buffer_target` raw texts from a streaming iterator before sequence construction."""
	target = buffer_target if buffer_target is not None else STREAM_SEQUENCE_BUFFER_TARGET
	if target <= 0:
		target = min_required
	target = max(target, min_required)
	texts: List[str] = []
	while len(texts) < target:
		try:
			raw = next(text_iter)
		except StopIteration:
			break
		text = str(raw).strip()
		if not text:
			continue
		texts.append(text)
	if len(texts) < min_required:
		raise ValueError(f"Requested at least {min_required} {label} but only collected {len(texts)} from streaming dataset.")
	return texts


def _texts_to_sequences_concat(texts: List[str], *, sequence_length: int) -> List[str]:
	"""Convert buffered raw texts into fixed-length sequences by concatenating tokens."""
	def _tokens_from_text(txt: str) -> List[str]:
		return str(txt).strip().split()
	sequences: List[str] = []
	buf_tokens: List[str] = []
	for raw_text in texts:
		tokens = _tokens_from_text(raw_text)
		if not tokens:
			continue
		buf_tokens.extend(tokens)
		while len(buf_tokens) >= sequence_length:
			sequences.append(" ".join(buf_tokens[:sequence_length]))
			buf_tokens = buf_tokens[sequence_length:]
	return sequences


def _texts_to_tail_sequences(texts: List[str], *, sequence_length: int) -> List[str]:
	"""Use the last `sequence_length` tokens from each text, skipping shorter entries."""
	def _tokens_from_text(txt: str) -> List[str]:
		return str(txt).strip().split()
	sequences: List[str] = []
	for raw_text in texts:
		tokens = _tokens_from_text(raw_text)
		if len(tokens) < sequence_length:
			continue
		sequences.append(" ".join(tokens[-sequence_length:]))
	return sequences
def _streaming_text_iterator(dataset_name: str, *, text_selector, seed_val: int, ds_config: str | None = None, split: str = "train", max_skip: int = 5000, **load_kwargs):
	"""Yield texts from a streaming dataset with a seed-driven random offset per restart."""
	rng_local = np.random.RandomState(seed_val)
	attempts = 0
	while attempts < 20:
		if ds_config is None:
			ds_iter = load_dataset(dataset_name, split=split, streaming=True, **load_kwargs)
		else:
			ds_iter = load_dataset(dataset_name, ds_config, split=split, streaming=True, **load_kwargs)
		it = iter(ds_iter)
		skip = int(rng_local.randint(0, max_skip + 1))
		for _ in range(skip):
			try:
				next(it)
			except StopIteration:
				it = iter(ds_iter)
				break
		for ex in it:
			try:
				yield text_selector(ex)
			except Exception:
				if isinstance(ex, dict) and len(ex) > 0:
					first_key = next(iter(ex.keys()))
					yield str(ex[first_key])
				else:
					yield ""
		attempts += 1
		seed_val += 1


def sample_splits(
	seed: int,
	train_total: int,
	eval_total: int,
	dataset: str = "ag_news",
	dataset_id: str | None = None,
	dataset_config: str | None = None,
	text_field: str | None = None,
	train_split: str = "train",
	test_split: str = "test",
	sequence_length: int = 128,
	return_sequence_iter: bool = False,
) -> Tuple[List[Example], List[Example], List[Example], List[Example]]:
	"""Load dataset and deterministically sample 4 splits matching temp.py protocol.

	Supported:
	- ag_news: uses 'text' field
	- xsum: uses 'document' field
	"""
	rng = np.random.RandomState(seed)

	def _tokens_from_text(txt: str) -> List[str]:
		return str(txt).strip().split()

	def _sequence_generator(text_iter):
		"""Yield sequences of exact length by concatenating consecutive training texts; discard overflow tokens."""
		buf_tokens: List[str] = []
		for raw_text in text_iter:
			tokens = _tokens_from_text(raw_text)
			if not tokens:
				continue
			buf_tokens.extend(tokens)
			if len(buf_tokens) >= sequence_length:
				yield " ".join(buf_tokens[:sequence_length])
				buf_tokens = []

	def _sequence_generator_tail(text_iter):
		"""Yield the last sequence_length tokens from each text (used for code datasets to avoid headers/imports)."""
		for raw_text in text_iter:
			tokens = _tokens_from_text(raw_text)
			if len(tokens) < sequence_length:
				continue
			yield " ".join(tokens[-sequence_length:])

	def _collect_sequences(seq_iter, needed: int) -> List[str]:
		out: List[str] = []
		for _ in range(needed):
			try:
				out.append(next(seq_iter))
			except StopIteration:
				break
		if len(out) < needed:
			raise ValueError(f"Requested {needed} sequences of length {sequence_length} but only produced {len(out)} from the train split. Consider lowering sequence_length or the requested totals.")
		return out

	if dataset_id:
		if dataset_config is None:
			ds = load_dataset(dataset_id)
		else:
			ds = load_dataset(dataset_id, dataset_config)
		if train_split not in ds:
			raise ValueError(f"Train split '{train_split}' not found for dataset_id={dataset_id}")
		train_ds = ds[train_split]
	elif dataset.lower() == "ag_news":
		ds = load_dataset("ag_news")
		train_ds = ds["train"]
	elif dataset.lower() == "xsum":
		ds = load_dataset("xsum")
		train_ds = ds["train"]
	elif dataset.lower() == "mtsamples":
		raw_texts = _load_mtsamples_raw_texts()
		buffer_sequences = _texts_to_sequences_concat(raw_texts, sequence_length=sequence_length)

		tm = train_total // 2
		em = eval_total // 2
		tn = train_total // 2
		en = eval_total // 2

		member_needed = tm + em
		nonmember_needed = tn + en
		total_needed = member_needed + nonmember_needed

		if len(buffer_sequences) < total_needed:
			raise ValueError(
				f"Requested {total_needed} sequences of length {sequence_length}, "
				f"but only produced {len(buffer_sequences)} from MTSamples."
			)

		rng_stream = np.random.RandomState(seed + 17)
		rng_stream.shuffle(buffer_sequences)

		member_texts = buffer_sequences[:member_needed]
		nonmember_texts = buffer_sequences[member_needed:member_needed + nonmember_needed]
		remaining_sequences = buffer_sequences[member_needed + nonmember_needed:]

		members = [Example(id=f"member_{i}", text=t, label=0) for i, t in enumerate(member_texts)]
		nonmembers = [Example(id=f"nonmember_{i}", text=t, label=0) for i, t in enumerate(nonmember_texts)]

		training_member = members[:tm]
		eval_member = members[tm: tm + em]
		training_nonmember = nonmembers[:tn]
		eval_nonmember = nonmembers[tn: tn + en]

		seq_out = iter(remaining_sequences)
		return (training_member, training_nonmember, eval_member, eval_nonmember, seq_out) if return_sequence_iter else (training_member, training_nonmember, eval_member, eval_nonmember)
	elif dataset.lower() == "wikitext":
		text_selector = lambda ex: ex.get("text", "")
		train_text_iter = _streaming_text_iterator("wikitext", text_selector=text_selector, seed_val=seed, ds_config="wikitext-103-raw-v1")
		tm = train_total // 2; em = eval_total // 2; tn = train_total // 2; en = eval_total // 2
		member_needed = tm + em
		nonmember_needed = tn + en
		total_needed = member_needed + nonmember_needed
		buffer_texts = _collect_streaming_texts(
			train_text_iter,
			min_required=total_needed * 2,
			label="wikitext texts",
		)
		buffer_sequences = _texts_to_sequences_concat(buffer_texts, sequence_length=sequence_length)
		if len(buffer_sequences) < total_needed:
			raise ValueError(f"Requested {total_needed} sequences of length {sequence_length} but only produced {len(buffer_sequences)} from wikitext stream.")
		rng_stream = np.random.RandomState(seed + 17)
		rng_stream.shuffle(buffer_sequences)
		member_texts = buffer_sequences[:member_needed]
		nonmember_texts = buffer_sequences[member_needed:member_needed + nonmember_needed]
		remaining_sequences = buffer_sequences[member_needed + nonmember_needed:]
		members = [Example(id=f"member_{i}", text=t, label=0) for i, t in enumerate(member_texts)]
		nonmembers = [Example(id=f"nonmember_{i}", text=t, label=0) for i, t in enumerate(nonmember_texts)]
		training_member = members[:tm]
		eval_member = members[tm: tm + em]
		training_nonmember = nonmembers[:tn]
		eval_nonmember = nonmembers[tn: tn + en]
		seq_out = iter(remaining_sequences)
		return (training_member, training_nonmember, eval_member, eval_nonmember, seq_out) if return_sequence_iter else (training_member, training_nonmember, eval_member, eval_nonmember)
	elif dataset.lower() in {"tokyotech-llm/swallow-code", "swallow-code"}:
		def pick_text(ex):
			for k in ("content", "text", "code", "docstring", "source"):
				if k in ex:
					return ex[k]
			return ex[next(iter(ex.keys()))]
		text_iter = _streaming_text_iterator(
			"codeparrot/codeparrot-clean",
			text_selector=pick_text,
			seed_val=seed,
		)
		tm = train_total // 2; em = eval_total // 2; tn = train_total // 2; en = eval_total // 2
		member_needed = tm + em
		nonmember_needed = tn + en
		seq_texts = _collect_streaming_texts(
			text_iter,
			min_required=(member_needed + nonmember_needed) * 2,
			label="swallow-code texts",
		)
		buffer_sequences = _texts_to_tail_sequences(seq_texts, sequence_length=sequence_length)
		if len(buffer_sequences) < member_needed + nonmember_needed:
			raise ValueError(f"Requested {member_needed + nonmember_needed} sequences of length {sequence_length} but only produced {len(buffer_sequences)} from swallow-code stream.")
		rng_stream = np.random.RandomState(seed + 17)
		rng_stream.shuffle(buffer_sequences)
		member_texts = buffer_sequences[:member_needed]
		nonmember_texts = buffer_sequences[member_needed:member_needed + nonmember_needed]
		remaining_sequences = buffer_sequences[member_needed + nonmember_needed:]
		members = [Example(id=f"member_{i}", text=t, label=0) for i, t in enumerate(member_texts)]
		nonmembers = [Example(id=f"nonmember_{i}", text=t, label=0) for i, t in enumerate(nonmember_texts)]
		training_member = members[:tm]
		eval_member = members[tm: tm + em]
		training_nonmember = nonmembers[:tn]
		eval_nonmember = nonmembers[tn: tn + en]
		seq_out = iter(remaining_sequences)
		return (training_member, training_nonmember, eval_member, eval_nonmember, seq_out) if return_sequence_iter else (training_member, training_nonmember, eval_member, eval_nonmember)

	if dataset_id:
		if text_field is not None:
			def _pick_text_generic(example_row):
				return example_row.get(text_field, "") if isinstance(example_row, dict) else ""
		else:
			def _pick_text_generic(example_row):
				if isinstance(example_row, dict):
					for k, v in example_row.items():
						if isinstance(v, (str, bytes)):
							return v
				return str(next(iter(example_row.values()))) if isinstance(example_row, dict) and example_row else ""
		text_getter = _pick_text_generic
	elif dataset.lower() == "ag_news":
		text_getter = lambda example_row: example_row.get("text", "")
	elif dataset.lower() == "xsum":
		text_getter = lambda example_row: example_row.get("document", "")
	else:
		text_getter = lambda example_row: str(next(iter(example_row.values()))) if isinstance(example_row, dict) and example_row else ""

	train_member_needed = max(0, train_total // 2)
	train_nonmember_needed = max(0, train_total // 2)
	eval_member_needed = max(0, eval_total // 2)
	eval_nonmember_needed = max(0, eval_total // 2)

	train_indices = rng.permutation(len(train_ds)).tolist()
	cursor = 0
	def _train_text_iter():
		nonlocal cursor
		while cursor < len(train_indices):
			idx = train_indices[cursor]
			cursor += 1
			row = train_ds[idx]
			yield text_getter(row) if isinstance(row, dict) else str(row)

	member_required = train_member_needed + eval_member_needed
	nonmember_required = train_nonmember_needed + eval_nonmember_needed
	seq_iter = _sequence_generator(_train_text_iter())
	pool = _collect_sequences(seq_iter, member_required + nonmember_required)
	rng_shuffle = np.random.RandomState(seed + 17)
	rng_shuffle.shuffle(pool)
	member_texts = pool[:member_required]
	nonmember_texts = pool[member_required:]
	member_examples = [Example(id=f"train_member_{i}", text=t, label=0) for i, t in enumerate(member_texts)]
	nonmember_examples = [Example(id=f"train_nonmember_{i}", text=t, label=0) for i, t in enumerate(nonmember_texts)]

	training_member = member_examples[:train_member_needed]
	eval_member = member_examples[train_member_needed: train_member_needed + eval_member_needed]
	training_nonmember = nonmember_examples[:train_nonmember_needed]
	eval_nonmember = nonmember_examples[train_nonmember_needed: train_nonmember_needed + eval_nonmember_needed]

	return (training_member, training_nonmember, eval_member, eval_nonmember, seq_iter) if return_sequence_iter else (training_member, training_nonmember, eval_member, eval_nonmember)


def _cleanup_iterable_dataset(ds) -> None:
	"""Release resources held by streaming datasets to avoid interpreter shutdown crashes."""
	try:
		ds._ex_iterable = None 
	except Exception:
		pass


def _extract_text(row, fields: List[str]) -> str:
	for f in fields:
		if isinstance(row, dict) and f in row and row[f]:
			return str(row[f])
	if isinstance(row, dict) and len(row) > 0:
		first = next(iter(row.values()))
		return str(first)
	return ""


def sample_domain_splits(
	seed: int,
	train_total: int,
	target_dataset: str,
	sequence_length: int = 128,
	val_total: int = 500,
) -> Tuple[List[Example], List[Example], List[str]]:
	"""Sample domain dataset splits for training model and LR training.

	Uses the domain dataset mapped to the target dataset (from _PREFIX_SOURCES).
	Returns (domain_member_examples, domain_nonmember_examples, domain_val_texts) where:
	- domain_member_examples: train_total // 2 (used to train the training model)
	- domain_nonmember_examples: train_total // 2 (not used to train the training model)
	- domain_val_texts: val_total sequences used for validation of domain-sourced models
	"""
	ds_name = target_dataset.lower()
	if ds_name == "mtsamples":
		raw_texts = _load_mtsamples_raw_texts()
		sequences = _texts_to_sequences_concat(raw_texts, sequence_length=sequence_length)

		member_needed = train_total // 2
		nonmember_needed = train_total // 2
		val_needed = max(0, int(val_total))
		total_needed = member_needed + nonmember_needed + val_needed

		if len(sequences) < total_needed:
			raise ValueError(
				f"Requested {total_needed} domain sequences but only produced {len(sequences)} from MTSamples."
			)

		rng_shuffle = np.random.RandomState(seed + 8888)
		rng_shuffle.shuffle(sequences)

		member_texts = sequences[:member_needed]
		nonmember_texts = sequences[member_needed:member_needed + nonmember_needed]
		val_texts = sequences[member_needed + nonmember_needed:member_needed + nonmember_needed + val_needed]

		domain_members = [Example(id=f"domain_member_{i}", text=t, label=1) for i, t in enumerate(member_texts)]
		domain_nonmembers = [Example(id=f"domain_nonmember_{i}", text=t, label=0) for i, t in enumerate(nonmember_texts)]

		return domain_members, domain_nonmembers, val_texts
	
	if ds_name not in _PREFIX_SOURCES:
		raise ValueError(f"No domain source configured for target dataset='{target_dataset}'.")
	cfg = _PREFIX_SOURCES[ds_name]
	rng = np.random.RandomState(seed + 7777)

	def _tokens_from_text(txt: str) -> List[str]:
		return str(txt).strip().split()

	def _sequence_generator(text_iter):
		buf_tokens: List[str] = []
		for raw_text in text_iter:
			tokens = _tokens_from_text(raw_text)
			if not tokens:
				continue
			buf_tokens.extend(tokens)
			while len(buf_tokens) >= sequence_length:
				yield " ".join(buf_tokens[:sequence_length])
				buf_tokens = buf_tokens[sequence_length:]

	def _collect_sequences_local(seq_iter, needed: int) -> List[str]:
		texts_out: List[str] = []
		while len(texts_out) < needed:
			try:
				txt = next(seq_iter)
			except StopIteration:
				break
			texts_out.append(txt)
		if len(texts_out) < needed:
			raise ValueError(f"Requested {needed} domain sequences but only produced {len(texts_out)}. Consider lowering sequence_length or requested totals.")
		return texts_out

	member_needed = train_total // 2
	nonmember_needed = train_total // 2
	val_needed = max(0, int(val_total))
	total_needed = member_needed + nonmember_needed + val_needed

	use_tail_sequences = ds_name in {"tokyotech-llm/swallow-code", "swallow-code"}
	text_label = f"{ds_name} texts"
	sequences: List[str] = []

	if cfg["type"] == "hf_dataset":
		is_streaming = bool(cfg.get("streaming", False))
		extra_kwargs = {k: v for k, v in cfg.items() if k not in {"type", "name", "config", "split", "text_fields", "streaming"}}
		if is_streaming:
			text_iter = _streaming_text_iterator(
				cfg["name"],
				text_selector=lambda row: _extract_text(row, cfg["text_fields"]),
				seed_val=seed + 1111,
				ds_config=cfg.get("config"),
				split=cfg.get("split", "train"),
				**extra_kwargs,
			)
			buffer_texts = _collect_streaming_texts(
				text_iter,
				min_required=min(STREAM_SEQUENCE_BUFFER_TARGET, total_needed * 2),
				label=text_label,
				buffer_target=max(STREAM_SEQUENCE_BUFFER_TARGET, total_needed * 2),
			)
			sequences = (
				_texts_to_tail_sequences(buffer_texts, sequence_length=sequence_length)
				if use_tail_sequences
				else _texts_to_sequences_concat(buffer_texts, sequence_length=sequence_length)
			)
		else:
			ds = _load_hf_split(
				cfg["name"],
				config=cfg.get("config"),
				split=cfg.get("split", "train"),
				streaming=False,
			)
			indices = rng.permutation(len(ds)).tolist()
			text_iter = (_extract_text(ds[idx], cfg["text_fields"]) for idx in indices)
			seq_iter = _sequence_generator(text_iter)
			sequences = _collect_sequences_local(seq_iter, total_needed)
	elif cfg["type"] in {"parquet", "json"}:
		data_files = cfg["data_files"]
		is_streaming = bool(cfg.get("streaming", False))
		if is_streaming:
			ds = load_dataset(cfg["type"], data_files={"train": data_files}, split="train", streaming=True)
			text_iter = (_extract_text(row, cfg["text_fields"]) for row in ds)
			buffer_texts = _collect_streaming_texts(
				text_iter,
				min_required=min(STREAM_SEQUENCE_BUFFER_TARGET, total_needed * 2),
				label=text_label,
				buffer_target=max(STREAM_SEQUENCE_BUFFER_TARGET, total_needed * 2),
			)
			sequences = (
				_texts_to_tail_sequences(buffer_texts, sequence_length=sequence_length)
				if use_tail_sequences
				else _texts_to_sequences_concat(buffer_texts, sequence_length=sequence_length)
			)
			_cleanup_iterable_dataset(ds)
		else:
			ds = load_dataset(cfg["type"], data_files={"train": data_files}, split="train", streaming=False)
			indices = rng.permutation(len(ds)).tolist()
			text_iter = (_extract_text(ds[idx], cfg["text_fields"]) for idx in indices)
			seq_iter = _sequence_generator(text_iter)
			sequences = _collect_sequences_local(seq_iter, total_needed)
	else:
		raise ValueError(f"Unsupported domain source type: {cfg['type']}")

	if len(sequences) < total_needed:
		raise ValueError(f"Requested {total_needed} domain sequences but only produced {len(sequences)}.")

	rng_shuffle = np.random.RandomState(seed + 8888)
	rng_shuffle.shuffle(sequences)

	member_texts = sequences[:member_needed]
	nonmember_texts = sequences[member_needed:member_needed + nonmember_needed]
	val_texts = sequences[member_needed + nonmember_needed:member_needed + nonmember_needed + val_needed]

	domain_members = [Example(id=f"domain_member_{i}", text=t, label=1) for i, t in enumerate(member_texts)]
	domain_nonmembers = [Example(id=f"domain_nonmember_{i}", text=t, label=0) for i, t in enumerate(nonmember_texts)]

	return domain_members, domain_nonmembers, val_texts


def sample_prefix_texts(
	seed: int,
	dataset: str,
	max_texts: int,
) -> List[str]:
	"""Sample prefix texts from a public dataset mapped to the target dataset's domain."""
	ds_name = dataset.lower()
	if ds_name not in _PREFIX_SOURCES:
		raise ValueError(f"No prefix source configured for dataset='{dataset}'.")
	cfg = _PREFIX_SOURCES[ds_name]
	rng = np.random.RandomState(seed + 1337)

	texts: List[str] = []
	try:
		if cfg["type"] == "hf_dataset":
			if cfg.get("streaming", False):
				text_iter = _streaming_text_iterator(
					cfg["name"],
					text_selector=lambda row: _extract_text(row, cfg["text_fields"]),
					seed_val=seed + 31337,
					ds_config=cfg.get("config"),
					split=cfg.get("split", "train"),
				)
				buffer_target = max(STREAM_SEQUENCE_BUFFER_TARGET, max_texts)
				texts = _collect_streaming_texts(
					text_iter,
					min_required=max_texts,
					label="prefix texts",
					buffer_target=buffer_target,
				)
				rng_stream = np.random.RandomState(seed + 31337)
				rng_stream.shuffle(texts)
			else:
				ds = _load_hf_split(
				cfg["name"],
				config=cfg.get("config"),
				split=cfg.get("split", "train"),
				streaming=False,
			)
				indices = rng.permutation(len(ds)).tolist()
				for idx in indices:
					row = ds[idx]
					txt = _extract_text(row, cfg["text_fields"])
					if txt:
						texts.append(txt)
					if len(texts) >= max_texts:
						break
		elif cfg["type"] in {"parquet", "json"}:
			data_files = cfg["data_files"]
			is_streaming = bool(cfg.get("streaming", False))
			if is_streaming:
				ds = load_dataset(cfg["type"], data_files={"train": data_files}, split="train", streaming=True)
				text_iter = (_extract_text(row, cfg["text_fields"]) for row in ds)
				buffer_target = max(STREAM_SEQUENCE_BUFFER_TARGET, max_texts)
				texts = _collect_streaming_texts(
					text_iter,
					min_required=max_texts,
					label="prefix texts",
					buffer_target=buffer_target,
				)
				rng_stream = np.random.RandomState(seed + 32323)
				rng_stream.shuffle(texts)
				_cleanup_iterable_dataset(ds)
			else:
				ds = load_dataset(cfg["type"], data_files={"train": data_files}, split="train", streaming=False)
				indices = rng.permutation(len(ds)).tolist()
				for idx in indices:
					row = ds[idx]
					txt = _extract_text(row, cfg["text_fields"])
					if txt:
						texts.append(txt)
					if len(texts) >= max_texts:
						break
		else:
			raise ValueError(f"Unsupported prefix source type: {cfg['type']}")
	finally:
		texts = texts[:max_texts]

	if len(texts) == 0:
		raise RuntimeError(f"Failed to load any prefix texts for dataset='{dataset}'.")
	return texts


def sample_validation_texts(
	seed: int,
	val_total: int,
	*,
	dataset: str = "ag_news",
	dataset_id: str | None = None,
	dataset_config: str | None = None,
	text_field: str | None = None,
	train_split: str = "train",
	test_split: str = "test",
	sequence_length: int = 128,
	sequence_iter=None,
) -> List[str]:
	"""Sample a validation corpus directly from the original dataset.

	This function is independent of training/eval subset sizes and does not reuse the
	previously sampled indices. For streaming datasets, it collects a deterministic
	window. For in-memory datasets, it shuffles deterministically and takes the first N.
	When `sequence_iter` is provided, it is consumed to avoid any overlap with train/eval.
	"""
	N = int(max(0, val_total))
	if N == 0:
		return []

	if sequence_iter is not None:
		texts: List[str] = []
		for _ in range(N):
			try:
				texts.append(next(sequence_iter))
			except StopIteration:
				break
		if len(texts) < N:
			raise ValueError(f"Requested {N} validation sequences of length {sequence_length} but only produced {len(texts)} from the shared iterator.")
		rng_seq = np.random.RandomState(seed + 999)
		rng_seq.shuffle(texts)
		return texts

	def _tokens_from_text(txt: str) -> List[str]:
		return str(txt).strip().split()

	def _sequence_generator(text_iter):
		buf_tokens: List[str] = []
		for raw_text in text_iter:
			tokens = _tokens_from_text(raw_text)
			if not tokens:
				continue
			buf_tokens.extend(tokens)
			if len(buf_tokens) >= sequence_length:
				yield " ".join(buf_tokens[:sequence_length])
				buf_tokens = []

	def _sequence_generator_tail(text_iter):
		for raw_text in text_iter:
			tokens = _tokens_from_text(raw_text)
			if not tokens or len(tokens) < sequence_length:
				continue
			yield " ".join(tokens[-sequence_length:])

	def _collect_sequences(seq_iter, needed: int) -> List[str]:
		out: List[str] = []
		for _ in range(needed):
			try:
				out.append(next(seq_iter))
			except StopIteration:
				break
		if len(out) < needed:
			raise ValueError(f"Requested {needed} validation sequences of length {sequence_length} but only produced {len(out)} from the train split.")
		return out

	ls = dataset.lower()
	if ls == "mtsamples":
		raw_texts = _load_mtsamples_raw_texts()
		buffer = _texts_to_sequences_concat(raw_texts, sequence_length=sequence_length)
		rng_stream = np.random.RandomState(seed + 1999)
		rng_stream.shuffle(buffer)
		if len(buffer) < N:
			raise ValueError(
				f"Requested {N} validation sequences but only produced {len(buffer)} from MTSamples."
			)
		return buffer[:N]
	if ls == "wikitext":
		text_selector = lambda ex: ex.get("text", "")
		text_iter = _streaming_text_iterator("wikitext", text_selector=text_selector, seed_val=seed + 999, ds_config="wikitext-103-raw-v1")
		buffer_texts = _collect_streaming_texts(
			text_iter,
			min_required=min(STREAM_SEQUENCE_BUFFER_TARGET, N * 2),
			label="wikitext val texts",
			buffer_target=max(STREAM_SEQUENCE_BUFFER_TARGET, N * 2),
		)
		buffer = _texts_to_sequences_concat(buffer_texts, sequence_length=sequence_length)
		rng_stream = np.random.RandomState(seed + 1999)
		rng_stream.shuffle(buffer)
		return buffer[:N]
	elif ls in {"tokyotech-llm/swallow-code", "swallow-code"}:
		pick_text = lambda ex: ex.get("content", ex.get("text", ex.get("code", ex.get("docstring", ex.get("source", "")))))
		text_iter = _streaming_text_iterator(
			"codeparrot/codeparrot-clean",
			text_selector=pick_text,
			seed_val=seed + 999,
		)
		buffer_texts = _collect_streaming_texts(
			text_iter,
			min_required=min(STREAM_SEQUENCE_BUFFER_TARGET, N * 2),
			label="swallow-code val texts",
			buffer_target=max(STREAM_SEQUENCE_BUFFER_TARGET, N * 2),
		)
		buffer = _texts_to_tail_sequences(buffer_texts, sequence_length=sequence_length)
		rng_stream = np.random.RandomState(seed + 1999)
		rng_stream.shuffle(buffer)
		return buffer[:N]

	if dataset_id:
		if dataset_config is None:
			ds = load_dataset(dataset_id)
		else:
			ds = load_dataset(dataset_id, dataset_config)
		if train_split not in ds:
			raise ValueError(f"Train split '{train_split}' not found for dataset_id={dataset_id}")
		train_ds = ds[train_split]
		if text_field is not None:
			def _text_getter(row):
				return row.get(text_field, "") if isinstance(row, dict) else ""
		else:
			def _text_getter(row):
				if isinstance(row, dict):
					for k, v in row.items():
						if isinstance(v, (str, bytes)):
							return v
				return str(next(iter(row.values()))) if isinstance(row, dict) and row else ""
	elif ls == "ag_news":
		ds = load_dataset("ag_news")
		train_ds = ds["train"]
		_text_getter = lambda r: r.get("text", "")
	elif ls == "xsum":
		ds = load_dataset("xsum")
		train_ds = ds["train"]
		_text_getter = lambda r: r.get("document", "")
	else:
		ds = load_dataset(ls)
		train_ds = ds["train"] if "train" in ds else next(iter(ds.values()))
		_text_getter = lambda r: str(next(iter(r.values()))) if isinstance(r, dict) and r else ""

	rng = np.random.RandomState(seed + 999)
	indices = rng.permutation(len(train_ds)).tolist()
	def _train_text_iter():
		for idx in indices:
			row = train_ds[idx]
			yield _text_getter(row) if isinstance(row, dict) else str(row)

	return _collect_sequences(_sequence_generator(_train_text_iter()), N)[:N]