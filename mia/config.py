from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf


AttackRefVariant = Literal["base", "distillation", "sft"]
DefenseType = Literal[
    "none",
    "output_perturbation",
    "bottom_k_smoothing",
]


@dataclass
class AttackConfig:
    dataset: str
    ref_variant: AttackRefVariant

    min_k_percent: float = 20.0
    seed: int = 42
    save_artifacts_path: str | None = None

    defense: DefenseType = "none"
    noise_std: float = 0.0
    risk_k_percent: float = 20.0
    smoothing_alpha: float = 0.8
    adaptive_beta: float = 2.0
    domain_dataset: str | None = None

    train_total: int = 20000
    eval_total: int = 20000

    epochs: int = 3
    batch_size: int = 16
    lr: float = 5e-5
    sequence_length: int = 128

    model_name: str = "gpt2"
    finetune_method: Literal["auto", "full", "lora"] = "auto"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None

    val_total: int = 500

    distil_max_prompts: int = 1000
    distil_completions: int = 1
    distil_max_new_tokens: int = 64
    distil_temperature: float = 0.9
    distil_top_p: float = 0.9
    distil_input_max_tokens: int = 128
    distil_train_epochs: int = 4
    distil_train_batch: int = 16
    distil_train_lr: float = 1e-4

    sft_train_epochs: int = 1
    sft_train_batch: int = 16
    sft_train_lr: float = 1e-4

    def validate(self) -> None:
        """Validate experiment settings before an attack run starts."""
        if not self.dataset or not self.dataset.strip():
            raise ValueError("dataset must be a non-empty string.")

        if self.ref_variant not in {"base", "distillation", "sft"}:
            raise ValueError(
                "ref_variant must be one of: base, distillation, sft."
            )
        


        if self.defense not in {
            "none",
            "output_perturbation",
            "bottom_k_smoothing",
        }:
            raise ValueError(
                "defense must be one of: none, output_perturbation, "
                "bottom_k_smoothing."
            )

        if self.finetune_method not in {"auto", "full", "lora"}:
            raise ValueError(
                "finetune_method must be one of: auto, full, lora."
            )

        if self.train_total <= 0:
            raise ValueError("train_total must be greater than zero.")

        if self.train_total % 2 != 0:
            raise ValueError(
                f"train_total must be even (got {self.train_total})."
            )

        if self.eval_total <= 0:
            raise ValueError("eval_total must be greater than zero.")

        if self.eval_total % 2 != 0:
            raise ValueError(
                f"eval_total must be even (got {self.eval_total})."
            )

        if self.val_total <= 0:
            raise ValueError("val_total must be greater than zero.")

        if self.epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if self.lr <= 0:
            raise ValueError("lr must be greater than zero.")

        if self.sequence_length < 2:
            raise ValueError(
                "sequence_length must be at least 2 for next-token scoring."
            )

        if self.noise_std < 0:
            raise ValueError("noise_std must be non-negative.")

        if not 0.0 < self.risk_k_percent <= 100.0:
            raise ValueError(
                "risk_k_percent must be greater than 0 and at most 100."
            )

        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError(
                "smoothing_alpha must be between 0 and 1."
            )

        if self.adaptive_beta < 0:
            raise ValueError("adaptive_beta must be non-negative.")

        if self.lora_r <= 0:
            raise ValueError("lora_r must be greater than zero.")

        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be greater than zero.")

        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(
                "lora_dropout must be at least 0 and less than 1."
            )

        if self.distil_max_prompts < 0:
            raise ValueError("distil_max_prompts must be non-negative.")

        if self.distil_completions <= 0:
            raise ValueError("distil_completions must be greater than zero.")

        if self.distil_max_new_tokens <= 0:
            raise ValueError(
                "distil_max_new_tokens must be greater than zero."
            )

        if self.distil_temperature <= 0:
            raise ValueError(
                "distil_temperature must be greater than zero."
            )

        if not 0.0 < self.distil_top_p <= 1.0:
            raise ValueError(
                "distil_top_p must be greater than 0 and at most 1."
            )

        if self.distil_input_max_tokens <= 0:
            raise ValueError(
                "distil_input_max_tokens must be greater than zero."
            )

        if self.distil_train_epochs <= 0:
            raise ValueError(
                "distil_train_epochs must be greater than zero."
            )

        if self.distil_train_batch <= 0:
            raise ValueError(
                "distil_train_batch must be greater than zero."
            )

        if self.distil_train_lr <= 0:
            raise ValueError(
                "distil_train_lr must be greater than zero."
            )

        if self.sft_train_epochs <= 0:
            raise ValueError(
                "sft_train_epochs must be greater than zero."
            )

        if self.sft_train_batch <= 0:
            raise ValueError(
                "sft_train_batch must be greater than zero."
            )

        if self.sft_train_lr <= 0:
            raise ValueError("sft_train_lr must be greater than zero.")

        if not 0.0 < self.min_k_percent <= 100.0:
            raise ValueError(
                 "min_k_percent must be greater than 0 and at most 100."
            )


def make_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for a single experiment."""
    ap = argparse.ArgumentParser(
        description=(
            "Run an MIA experiment using direct CLI arguments or a YAML "
            "configuration file."
        )
    )

    ap.add_argument(
    "--min-k-percent",
    type=float,
    default=20.0,
    help=(
        "Percentage of lowest correct-token log probabilities "
        "used by the Min-K% membership inference attack."
    ),
)

    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a YAML configuration file. When provided, the YAML "
            "values are used for the experiment."
        ),
    )

    ap.add_argument(
        "--dataset",
        choices=[
            "ag_news",
            "xsum",
            "wikitext",
            "mtsamples",
            "tokyotech-llm/swallow-code",
        ],
        default="ag_news",
    )
    ap.add_argument(
        "--ref-variant",
        choices=["base", "distillation", "sft"],
        default="base",
    )
    ap.add_argument(
        "--target-model",
        choices=[
            "gpt2",
            "EleutherAI/gpt-j-6B",
            "meta-llama/Llama-2-7b-hf",
            "codellama/CodeLlama-7b-hf",
        ],
        default="gpt2",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-total", type=int, default=8000)
    ap.add_argument("--eval-total", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument(
        "--sequence-length",
        type=int,
        default=128,
        help=(
            "Exact whitespace-token length for each training/evaluation "
            "example. Long samples are truncated and short consecutive "
            "samples may be concatenated."
        ),
    )
    ap.add_argument(
        "--val-total",
        type=int,
        default=500,
        help=(
            "Number of validation examples sampled independently from the "
            "training and evaluation subsets."
        ),
    )

    ap.add_argument(
        "--defense",
        choices=[
            "none",
            "output_perturbation",
            "bottom_k_smoothing",
        ],
        default="none",
        help="Defense applied during attack scoring and utility evaluation.",
    )
    ap.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help=(
            "Standard deviation of output noise used by the "
            "output_perturbation defense."
        ),
    )
    ap.add_argument(
        "--risk-k-percent",
        type=float,
        default=20.0,
        help=(
            "Percentage of highest-risk token positions considered by the "
            "adaptive defense logic."
        ),
    )
    ap.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.8,
        help="Smoothing strength used by bottom_k_smoothing.",
    )
    ap.add_argument(
        "--adaptive-beta",
        type=float,
        default=2.0,
        help="Adaptive scaling factor used by the defense.",
    )
    ap.add_argument(
        "--domain-dataset",
        type=str,
        default=None,
        help=(
            "Optional domain dataset override. When omitted, the target "
            "dataset is used by the existing domain-sampling logic."
        ),
    )

    ap.add_argument(
        "--finetune-method",
        choices=["auto", "full", "lora"],
        default="auto",
    )
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--distil-max-prompts", type=int, default=1000)
    ap.add_argument("--distil-completions", type=int, default=1)
    ap.add_argument("--distil-max-new-tokens", type=int, default=64)
    ap.add_argument("--distil-temperature", type=float, default=0.9)
    ap.add_argument("--distil-top-p", type=float, default=0.9)
    ap.add_argument("--distil-input-max-tokens", type=int, default=128)
    ap.add_argument("--distil-train-epochs", type=int, default=4)
    ap.add_argument("--distil-train-batch", type=int, default=16)
    ap.add_argument("--distil-train-lr", type=float, default=1e-4)

    ap.add_argument("--sft-train-epochs", type=int, default=1)
    ap.add_argument("--sft-train-batch", type=int, default=16)
    ap.add_argument("--sft-train-lr", type=float, default=1e-4)

    ap.add_argument(
        "--save-artifacts-path",
        type=str,
        default=None,
        help=(
            "Path to save models and data for later use with "
            "verification_harness.py. Disabled by default."
        ),
    )

    return ap


def load_attack_config_from_yaml(path: str) -> AttackConfig:
    """Load and validate AttackConfig from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")

    oc = OmegaConf.load(str(cfg_path))
    data = OmegaConf.to_container(oc, resolve=True) or {}

    if not isinstance(data, dict):
        raise ValueError(
            "YAML root must be a mapping of AttackConfig fields."
        )

    cfg = AttackConfig(**data)
    cfg.validate()
    return cfg