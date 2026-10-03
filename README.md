<div align="center">

# Advantage Estimation Diagnostics

Reproducing a GSM8K comparison of GRPO, VinePPO, SPO-int5, and PPO advantage estimators.

[Original SPO project](https://github.com/AIFrameResearch/SPO) | [Original SPO paper](https://arxiv.org/abs/2505.23564)

</div>

## Overview

This repository extends the original SPO code with a diagnostic experiment on fixed GSM8K trajectories. It computes two independent Monte Carlo reference advantages, estimates advantages with GRPO, VinePPO, SPO-int5, and a trained PPO Critic, then evaluates overall risk (`R`), partition approximation error (`A`), and `E = R - A`.

The commands below reproduce the **128-question, one-round** experiment. They compare advantage estimators on the same trajectories; they do not train four RL policies or measure downstream policy accuracy. Run the commands from the repository root on Linux with a CUDA GPU. The final metric calculations are CPU-only.

## Getting Started

### 1. Install dependencies

The pinned requirements include Linux wheels for PyTorch, FlashAttention, and vLLM 0.4.0.post1. Python 3.10 and a compatible CUDA installation are required.

```bash
conda create -n spo-diagnostic python=3.10
conda activate spo-diagnostic
pip install -r requirements.txt

export PYTHONPATH="$PWD/src:$PWD/scripts:${PYTHONPATH:-}"
```

### 2. Set paths and obtain the base model

Choose a writable output directory with enough space for Critic checkpoints and Monte Carlo rollout audits. Keep the same base-model checkpoint for every stage.

```bash
export PROJECT_ROOT="$PWD"
export RUN_DIR="/path/to/experiment-output"
export DATASET_PATH="$PROJECT_ROOT/data/gsm8k"
export MODEL_PATH="$RUN_DIR/models/rho-1b-sft-GSM8K"
export SELECTION_DIR="$RUN_DIR/gsm8k_validation_selection"
export CRITIC_DIR="$RUN_DIR/critic_models/rho1b_gsm8k_fp32amp_lr3e6"

mkdir -p "$RUN_DIR" "$RUN_DIR/models"
huggingface-cli download realtreetune/rho-1b-sft-GSM8K --local-dir "$MODEL_PATH"
```

If the model is already available locally, set `MODEL_PATH` to that directory and skip the download. Record the exact model revision or file hashes for a published reproduction.

### 3. Prepare GSM8K

The repository's dataset preparation script populates `data/`. The selection script requires `DATASET_PATH` to be a Hugging Face `DatasetDict` saved with `save_to_disk()` and to contain a `validation` split. Check that condition before continuing; it will not substitute `train` or `test`.

```bash
bash scripts/download_and_prepare_dataset.sh

python -c 'from datasets import load_from_disk; import os; ds = load_from_disk(os.environ["DATASET_PATH"]); print(ds.keys()); assert "validation" in ds'
```

### 4. Select diagnostic and Critic questions

Select 373 validation questions with seed 42. The first 128 form the diagnostic pool; the remaining 245 form a question-disjoint Critic pool. The 32-question pilot is a subset of the diagnostic pool and is not used in the full run.

```bash
python scripts/select_gsm8k_validation.py \
  --dataset-path "$DATASET_PATH" \
  --split validation \
  --num-samples 373 \
  --diagnostic-size 128 \
  --pilot-size 32 \
  --seed 42 \
  --output-dir "$SELECTION_DIR"
```

## Generate Trajectories

### 5. Generate the 128 diagnostic trajectory groups

Each question receives eight vLLM rollouts. The script selects one saved rollout per question for the reference and method comparisons. The prompt and stop strings below match the GSM8K project configuration. Keep token/text mismatch checks enabled.

```bash
python scripts/generate_diagnostic_trajectories_vllm.py \
  --dataset-path "$SELECTION_DIR/diagnostic" \
  --model-name-or-path "$MODEL_PATH" \
  --output-path "$RUN_DIR/diagnostic_trajectories_128_vllm.jsonl" \
  --group-size 8 \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-new-tokens 1024 \
  --prompt-template $'[MATH_TASK] Problem:\n{query}\n\nSolution:' \
  --stop-text $'\n\n\nProblem:' \
  --seed 42 \
  --logprobs 1
```

The adjacent `.meta.json` file records the sampling and reward protocols. All subsequent commands consume this exact trajectory file.

### 6. Prepare the project partitions

This step scores the fixed trajectories with the actor and applies the repository's VinePPO and SPO-int5 partition logic. The diagnostic SPO estimator does not use the SPO probability mask.

```bash
python scripts/prepare_method_partitions.py \
  --trajectories-path "$RUN_DIR/diagnostic_trajectories_128_vllm.jsonl" \
  --output-path "$RUN_DIR/method_partitions_project_v2.jsonl"
```

## Estimate Advantages

### 7. Compute two independent reference groups

Each group samples eight continuations per token prefix. `--rounds 1` fixes the protocol used for the reported 128-question result. The project reward assigns 1 to a correct answer, 0 to an incorrect or unfinished answer, and -2 for the project's multiple-answer penalty.

```bash
python scripts/compute_token_reference_advantages_vllm.py \
  --trajectories-path "$RUN_DIR/diagnostic_trajectories_128_vllm.jsonl" \
  --output-path "$RUN_DIR/token_reference_advantages_128.jsonl" \
  --mc-samples 8 \
  --rounds 1 \
  --backend engine \
  --max-pending-requests 16 \
  --max-num-seqs 256 \
  --reward-mode project
```

### 8. Compute GRPO, VinePPO, and SPO-int5 estimates

The project estimator protocol uses nine Monte Carlo completions per request. The saved partition file and trajectory file must correspond to one another.

```bash
python scripts/compute_method_advantages_vllm.py \
  --trajectories-path "$RUN_DIR/diagnostic_trajectories_128_vllm.jsonl" \
  --partitions-path "$RUN_DIR/method_partitions_project_v2.jsonl" \
  --output-path "$RUN_DIR/method_advantages_128.jsonl" \
  --mc-samples 9 \
  --rounds 1 \
  --backend engine \
  --max-pending-requests 16 \
  --max-num-seqs 256
```

## Train the PPO Critic

### 9. Generate Critic rollouts

The Critic pool is disjoint from the diagnostic pool. Generate 16 rollouts for each of its 245 questions. Use the same model, prompt, sampling parameters, and reward protocol, with a separate generation seed.

```bash
python scripts/generate_diagnostic_trajectories_vllm.py \
  --dataset-path "$SELECTION_DIR/critic" \
  --model-name-or-path "$MODEL_PATH" \
  --output-path "$RUN_DIR/critic_rollouts_245.jsonl" \
  --group-size 16 \
  --temperature 0.6 \
  --top-p 0.9 \
  --max-new-tokens 1024 \
  --prompt-template $'[MATH_TASK] Problem:\n{query}\n\nSolution:' \
  --stop-text $'\n\n\nProblem:' \
  --seed 1000042 \
  --logprobs 1
```

### 10. Prepare question-disjoint Critic training and validation data

The preparation script verifies question separation, reward agreement, sampling metadata, and response text/token consistency. It retains sampled vLLM response token IDs for the Critic inputs.

```bash
python scripts/prepare_diagnostic_critic_data.py \
  --rollouts-path "$RUN_DIR/critic_rollouts_245.jsonl" \
  --selection-dir "$SELECTION_DIR" \
  --diagnostic-metadata-path "$RUN_DIR/diagnostic_trajectories_128_vllm.meta.json" \
  --output-dir "$RUN_DIR/critic_training_data_v2" \
  --validation-fraction 0.2 \
  --split-seed 42
```

### 11. Train and select the Critic

The Critic is evaluated once per training round. Training stops after five consecutive rounds without a validation-MSE improvement greater than `1e-6`, or at 100 rounds. The best checkpoint is saved under `best/`; the two most recent round checkpoints are retained for resuming.

```bash
python scripts/train_diagnostic_critic.py \
  --data-dir "$RUN_DIR/critic_training_data_v2" \
  --model-name-or-path "$MODEL_PATH" \
  --output-dir "$CRITIC_DIR" \
  --max-rounds 100 \
  --epochs-per-round 2 \
  --learning-rate 3e-6 \
  --batch-size 8 \
  --gradient-accumulation-steps 8 \
  --warmup-ratio 0.03 \
  --cliprange-value 0.2 \
  --max-grad-norm 1.0 \
  --gradient-checkpointing \
  --patience 5 \
  --min-delta 1e-6 \
  --seed 42
```

### 12. Compute PPO Critic advantages on the diagnostic trajectories

Use the best Critic checkpoint. PPO uses `gamma = 1`, `lambda = 1`, the stored terminal reward, and a zero terminal value. This stage uses only the selected diagnostic rollout for each question.

```bash
python scripts/compute_ppo_critic_advantages.py \
  --trajectories-path "$RUN_DIR/diagnostic_trajectories_128_vllm.jsonl" \
  --critic-checkpoint-dir "$CRITIC_DIR/best" \
  --model-name-or-path "$MODEL_PATH" \
  --output-path "$RUN_DIR/ppo_critic_advantages_128.jsonl" \
  --batch-size 1 \
  --gamma 1.0 \
  --lam 1.0
```

## Evaluation

### 13. Evaluate GRPO, VinePPO, and SPO-int5

`reference-target` evaluates the saved reference target without silently changing its prefix or terminal conventions. `project` limits the comparison to the token positions used by the project estimators; a terminal reference EOS may be omitted. Bootstrap resampling uses questions as the independent unit.

```bash
python scripts/compute_diagnostic_metrics.py \
  --references-path "$RUN_DIR/token_reference_advantages_128.jsonl" \
  --estimates-path "$RUN_DIR/method_advantages_128.jsonl" \
  --output-dir "$RUN_DIR/final_diagnostic_report_128" \
  --comparison reference-target \
  --token-scope project \
  --bootstrap 5000 \
  --confidence 0.95 \
  --seed 42
```

### 14. Evaluate PPO on the same project token positions

The project-estimates file supplies the token mapping used in the preceding evaluation. PPO has a singleton-token partition, so its partition approximation error `A` is zero by definition.

```bash
python scripts/compute_ppo_diagnostic_metrics.py \
  --references-path "$RUN_DIR/token_reference_advantages_128.jsonl" \
  --ppo-advantages-path "$RUN_DIR/ppo_critic_advantages_128.jsonl" \
  --project-estimates-path "$RUN_DIR/method_advantages_128.jsonl" \
  --output-dir "$RUN_DIR/final_ppo_diagnostic_report_128_project" \
  --token-scope project \
  --bootstrap 5000 \
  --confidence 0.95 \
  --seed 42
```

### 15. Inspect the reports

Both report directories contain `summary.json`, `summary.csv`, and `per_question.jsonl`. Compare the question-weighted `R` values for the four methods; smaller values indicate lower diagnostic risk. The reported `A` values measure partition approximation error and should not be interpreted as PPO training accuracy.

```bash
cat "$RUN_DIR/final_diagnostic_report_128/summary.csv"
cat "$RUN_DIR/final_ppo_diagnostic_report_128_project/summary.csv"
```

For a repeat experiment with more Monte Carlo rounds, change `--rounds` in **both** advantage-estimation commands, write to new output filenames, and rerun both metric commands. The fixed trajectories, partitions, and PPO Critic advantages can be reused. Output directories must be new; scripts that reject an existing output path should also use fresh filenames.

## Acknowledgment

This diagnostic code builds on [Segment Policy Optimization](https://github.com/AIFrameResearch/SPO), which in turn acknowledges [VinePPO](https://github.com/McGill-NLP/VinePPO). Please retain the original project attribution and license when redistributing this repository.

## Citation

If you use the underlying SPO implementation, cite the original paper:

```bibtex
@misc{guo2025segmentpolicyoptimizationeffective,
  title={Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models},
  author={Yiran Guo and Lijie Xu and Jie Liu and Dan Ye and Shuang Qiu},
  year={2025},
  eprint={2505.23564},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2505.23564}
}
```

## License

See [LICENSE](LICENSE) for the license inherited from the original repository.
