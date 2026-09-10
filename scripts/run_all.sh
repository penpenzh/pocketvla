#!/usr/bin/env bash
# Per-size pipeline:
#   sft_<size>.yaml        -> SFT train + eval        (outputs_sft_<size>/)
#   grpo_<size>.yaml       -> GRPO RL + final eval    (outputs_grpo_<size>/)
#   ood evaluation         -> both stages' checkpoints on the OOD set
#                             (outputs_sft_<size>/eval_ood, outputs_grpo_<size>/eval)
# Ablation variants: *_noaux.yaml (no aux heads).
# Stages whose artifacts already exist are SKIPPED (completion detection), so the
# script is safe to re-run after an interruption.
# The dataset must already exist: run "python3 -m pvla collect" once.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
    pairs=(configs/sft_3m.yaml configs/grpo_3m.yaml configs/sft_3m_noaux.yaml)
else
    pairs=("$@")
fi

# completion markers
sft_done()   { compgen -G "$1/train/pocketvla-*/model.safetensors" >/dev/null; }
sft_eval_done()   { compgen -G "$1/eval/eval_report.json" >/dev/null; }
grpo_done()  { compgen -G "$1/pocketvla-*-grpo/model.safetensors" >/dev/null; }
grpo_eval_done()  { compgen -G "$1/eval/eval_report.json" >/dev/null; }
ood_done()   { compgen -G "$1/eval_ood/eval_report.json" >/dev/null; }

for cfg in "${pairs[@]}"; do
    name="$(basename "${cfg}" .yaml)"          # e.g. sft_3m / grpo_3m / sft_3m_noaux
    size="${name#sft_}"; size="${size#grpo_}"  # 3m / 20m
    sft_yaml="configs/sft_${size}.yaml"
    grpo_yaml="configs/grpo_${size}.yaml"
    echo ">>> PocketVLA ${cfg} (size=${size})"

    if [[ "${cfg}" == configs/sft_* ]]; then
        sft_root="outputs_sft_${size}"
        if sft_done "outputs_sft_${size}"; then
            echo "    SFT checkpoint already exists — skip train"
        else
            python3 -m pvla train --config "${cfg}" ${PVLA_EXTRA_ARGS:-}
        fi
        if sft_eval_done "outputs_sft_${size}"; then
            echo "    SFT eval already done — skip"
        else
            python3 -m pvla evaluate --config "${cfg}" ${PVLA_EXTRA_ARGS:-}
        fi
        if ood_done "outputs_sft_${size}"; then
            echo "    SFT OOD eval already done — skip"
        else
            echo "    OOD evaluation (random 15 GIFs)..."
            python3 -m pvla.eval_ood --config "${cfg}" --gifs 15 ${PVLA_EXTRA_ARGS:-}
        fi

    elif [[ "${cfg}" == configs/grpo_* ]]; then
        if grpo_done "outputs_grpo_${size}"; then
            echo "    GRPO checkpoint already exists — skip RL (delete outputs_grpo_${size} to redo)"
        else
            python3 -m pvla grpo --config "${cfg}" ${PVLA_EXTRA_ARGS:-}
        fi
        if grpo_eval_done "outputs_grpo_${size}"; then
            echo "    GRPO eval already done — skip (GRPO runs its final eval internally)"
        fi
        # OOD generalization check on the GRPO model
        if ood_done "outputs_grpo_${size}"; then
            echo "    GRPO OOD eval already done — skip"
        else
            echo "    GRPO OOD evaluation (random 15 GIFs)..."
            python3 -m pvla.eval_ood --config "${cfg}" \
                --ckpt "outputs_grpo_${size}/pocketvla-${size}-grpo" --gifs 15 ${PVLA_EXTRA_ARGS:-}
        fi

    else
        echo "unsupported yaml (expected configs/sft_*.yaml or configs/grpo_*.yaml): ${cfg}" >&2
        exit 1
    fi
done
