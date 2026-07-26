from __future__ import annotations

import csv
from pathlib import Path

from .attack import run_attack
from .config import (
    AttackConfig,
    load_attack_config_from_yaml,
    make_arg_parser,
)


def append_result_csv(row: dict):
    output_path = Path(__file__).parent / "results.csv"
    file_exists = output_path.exists()

    with output_path.open("a", newline="") as file_obj:
        writer = csv.writer(file_obj)

        if not file_exists:
            writer.writerow([
                "dataset",
                "target_model",
                "reference_variant",
                "defense",
                "attack_variant",
                "metric",
                "value",
                "seed",
            ])

        dataset = row["dataset"]
        target_model = row.get("target_model", "gpt2")
        ref_variant = row["ref_variant"]
        seed = row["seed"]
        defense_name = row.get("defense", "none")
        min_k_percent = float(row.get("min_k_percent", 20.0))
        min_k_variant = f"min_k_{min_k_percent:g}_percent"

        # EZ-MIA metrics
        writer.writerow([
            dataset,
            target_model,
            ref_variant,
            defense_name,
            "ez_ratio",
            "AUC",
            f"{row['auc']:.6f}",
            seed,
        ])

        if "tpr_at_fpr_0.01" in row:
            writer.writerow([
                dataset,
                target_model,
                ref_variant,
                defense_name,
                "ez_ratio",
                "TPR@1%FPR",
                f"{row['tpr_at_fpr_0.01']:.3f}",
                seed,
            ])

        writer.writerow([
            dataset,
            target_model,
            ref_variant,
            defense_name,
            "ez_ratio",
            "TPR@0.1%FPR",
            f"{row['tpr_at_fpr_0.001']:.3f}",
            seed,
        ])

        ez_classification_metrics = {
            "Accuracy@0.1%FPR": row.get(
                "accuracy_at_fpr_0.001"
            ),
            "Precision@0.1%FPR": row.get(
                "precision_at_fpr_0.001"
            ),
            "Recall@0.1%FPR": row.get(
                "recall_at_fpr_0.001"
            ),
            "F1@0.1%FPR": row.get(
                "f1_at_fpr_0.001"
            ),
            "Threshold@0.1%FPR": row.get(
                "threshold_at_fpr_0.001"
            ),
            "Actual FPR@0.1%FPR": row.get(
                "actual_fpr_at_fpr_0.001"
            ),
        }

        for metric_name, metric_value in (
            ez_classification_metrics.items()
        ):
            if metric_value is None:
                continue

            writer.writerow([
                dataset,
                target_model,
                ref_variant,
                defense_name,
                "ez_ratio",
                metric_name,
                f"{float(metric_value):.8f}",
                seed,
            ])

        # Min-K% metrics
        writer.writerow([
            dataset,
            target_model,
            ref_variant,
            defense_name,
            min_k_variant,
            "AUC",
            f"{row['min_k_auc']:.6f}",
            seed,
        ])

        if "min_k_tpr_at_fpr_0.01" in row:
            writer.writerow([
                dataset,
                target_model,
                ref_variant,
                defense_name,
                min_k_variant,
                "TPR@1%FPR",
                f"{row['min_k_tpr_at_fpr_0.01']:.3f}",
                seed,
            ])

        writer.writerow([
            dataset,
            target_model,
            ref_variant,
            defense_name,
            min_k_variant,
            "TPR@0.1%FPR",
            f"{row['min_k_tpr_at_fpr_0.001']:.3f}",
            seed,
        ])

        min_k_classification_metrics = {
            "Accuracy@0.1%FPR": row.get(
                "min_k_accuracy_at_fpr_0.001"
            ),
            "Precision@0.1%FPR": row.get(
                "min_k_precision_at_fpr_0.001"
            ),
            "Recall@0.1%FPR": row.get(
                "min_k_recall_at_fpr_0.001"
            ),
            "F1@0.1%FPR": row.get(
                "min_k_f1_at_fpr_0.001"
            ),
            "Threshold@0.1%FPR": row.get(
                "min_k_threshold_at_fpr_0.001"
            ),
            "Actual FPR@0.1%FPR": row.get(
                "min_k_actual_fpr_at_fpr_0.001"
            ),
        }

        for metric_name, metric_value in (
            min_k_classification_metrics.items()
        ):
            if metric_value is None:
                continue

            writer.writerow([
                dataset,
                target_model,
                ref_variant,
                defense_name,
                min_k_variant,
                metric_name,
                f"{float(metric_value):.8f}",
                seed,
            ])

        # Utility metrics
        utility_metrics = {
            "Baseline Perplexity": row.get(
                "baseline_perplexity"
            ),
            "Defended Perplexity": row.get(
                "defended_perplexity"
            ),
            "Perplexity Change Percent": row.get(
                "perplexity_change_percent"
            ),
            "Top-1 Agreement": row.get(
                "top1_agreement"
            ),
            "JS Divergence": row.get(
                "js_divergence"
            ),
            "Utility Token Count": row.get(
                "utility_token_count"
            ),
            "Utility Example Count": row.get(
                "utility_example_count"
            ),
        }

        for metric_name, metric_value in utility_metrics.items():
            if metric_value is None:
                continue

            writer.writerow([
                dataset,
                target_model,
                ref_variant,
                defense_name,
                "utility",
                metric_name,
                f"{float(metric_value):.8f}",
                seed,
            ])


def main():
    ap = make_arg_parser()
    args = ap.parse_args()

    if getattr(args, "config", None):
        cfg = load_attack_config_from_yaml(args.config)
        res = run_attack(cfg)

        print(
            f"{cfg.dataset}, "
            f"model={cfg.model_name}, "
            f"ref={cfg.ref_variant}, "
            f"defense={cfg.defense}, "
            f"Min-K={cfg.min_k_percent:g}%, "
            f"EZ-MIA AUC={res['auc']:.6f}, "
            f"EZ-MIA TPR@0.1%FPR="
            f"{res['tpr_at_fpr_0.001']:.3f}, "
            f"EZ Accuracy@0.1%FPR="
            f"{res['accuracy_at_fpr_0.001']:.4f}, "
            f"EZ Precision@0.1%FPR="
            f"{res['precision_at_fpr_0.001']:.4f}, "
            f"EZ Recall@0.1%FPR="
            f"{res['recall_at_fpr_0.001']:.4f}, "
            f"EZ F1@0.1%FPR="
            f"{res['f1_at_fpr_0.001']:.4f}, "
            f"Min-K% AUC={res['min_k_auc']:.6f}, "
            f"Min-K% TPR@0.1%FPR="
            f"{res['min_k_tpr_at_fpr_0.001']:.3f}, "
            f"Min-K Accuracy@0.1%FPR="
            f"{res['min_k_accuracy_at_fpr_0.001']:.4f}, "
            f"Min-K Precision@0.1%FPR="
            f"{res['min_k_precision_at_fpr_0.001']:.4f}, "
            f"Min-K Recall@0.1%FPR="
            f"{res['min_k_recall_at_fpr_0.001']:.4f}, "
            f"Min-K F1@0.1%FPR="
            f"{res['min_k_f1_at_fpr_0.001']:.4f}, "
            f"Baseline PPL="
            f"{res['baseline_perplexity']:.4f}, "
            f"Defended PPL="
            f"{res['defended_perplexity']:.4f}, "
            f"PPL Change="
            f"{res['perplexity_change_percent']:+.2f}%, "
            f"Top-1 Agreement="
            f"{res['top1_agreement']:.4f}, "
            f"JS Divergence="
            f"{res['js_divergence']:.8f}"
        )

        append_result_csv(res)
        return

    cfg = AttackConfig(
        dataset=args.dataset,
        ref_variant=args.ref_variant,
        seed=args.seed,
        save_artifacts_path=getattr(
            args,
            "save_artifacts_path",
            None,
        ),
        train_total=args.train_total,
        eval_total=args.eval_total,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sequence_length=args.sequence_length,
        model_name=args.target_model,
        finetune_method=args.finetune_method,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        val_total=args.val_total,
        defense=args.defense,
        noise_std=args.noise_std,
        min_k_percent=args.min_k_percent,
        risk_k_percent=args.risk_k_percent,
        smoothing_alpha=args.smoothing_alpha,
        adaptive_beta=args.adaptive_beta,
        domain_dataset=args.domain_dataset,
        distil_max_prompts=args.distil_max_prompts,
        distil_completions=args.distil_completions,
        distil_max_new_tokens=args.distil_max_new_tokens,
        distil_temperature=args.distil_temperature,
        distil_top_p=args.distil_top_p,
        distil_input_max_tokens=args.distil_input_max_tokens,
        distil_train_epochs=args.distil_train_epochs,
        distil_train_batch=args.distil_train_batch,
        distil_train_lr=args.distil_train_lr,
        sft_train_epochs=args.sft_train_epochs,
        sft_train_batch=args.sft_train_batch,
        sft_train_lr=args.sft_train_lr,
    )

    cfg.validate()

    res = run_attack(cfg)

    print(
        f"{args.dataset}, "
        f"model={cfg.model_name}, "
        f"ref={cfg.ref_variant}, "
        f"defense={cfg.defense}, "
        f"Min-K={cfg.min_k_percent:g}%, "
        f"AUC={res['auc']:.6f}, "
        f"TPR@0.1%FPR={res['tpr_at_fpr_0.001']:.3f}"
    )

    append_result_csv(res)


if __name__ == "__main__":
    main()