# 128-Question Diagnostic Experiment Data

This directory contains the data used in the 128-question
advantage-estimation diagnostic experiment. Each JSONL file has a
corresponding metadata sidecar.

## Files

| Data file | Metadata file | Contents |
|---|---|---|
| `diagnostic_trajectories_128_vllm.jsonl` | `diagnostic_trajectories_128_vllm.meta.json` | Sampled diagnostic trajectories. |
| `token_reference_advantages_128_r3.jsonl` | `token_reference_advantages_128_r3.meta.json` | Token-level reference advantages estimated with Monte Carlo continuations. |
| `method_advantages_128_r3.jsonl` | `method_advantages_128_r3.meta.json` | Advantage estimates for GRPO, VinePPO, and SPO-int5. |
| `ppo_critic_advantages_128.jsonl` | `ppo_critic_advantages_128.meta.json` | Token-level PPO advantages computed with the trained Critic. |

Each JSONL file contains one JSON record per non-empty line. Records across
files are associated by `row_index`. Use the metadata sidecars to inspect
the recorded generation and estimation settings, record counts, input
fingerprints, and completion status.

## Experiment Overview

The experiment uses 128 selected questions from the GSM8K validation set.
Diagnostic trajectories were sampled from
`realtreetune/rho-1b-sft-GSM8K` with vLLM. The recorded trajectory protocol
used a group size of 8, temperature 0.6, top-p 0.9, a maximum of 1,024 new
tokens, and seed 42.

Reference advantages were estimated from two independent groups of Monte
Carlo continuations. Method advantages were estimated for GRPO, VinePPO,
and SPO-int5. The PPO advantages were computed using a trained Critic.

The exact parameters for each output—including the sampler version,
number of Monte Carlo samples, number of rounds, reward protocol, and PPO
settings—are recorded in the corresponding metadata sidecar. Consult those
files when reproducing or interpreting a result.

## Data Alignment

Match records using `row_index`. Before combining files, verify that they
contain the same question set and that token-level arrays align with the
trajectory's saved token mapping. Do not assume the final token is EOS or
remove it without checking the saved indices.

The advantage files represent estimates at their documented processing
stage. In particular, they should not be assumed to include any
batch-level advantage whitening performed later by the project's PPO
trainer. Use the metadata and the repository's evaluation scripts to
confirm which stage a given file represents.

## Scope and Limitations

This release contains the four JSONL data files and their metadata
sidecars. It does not include model weights, the trained Critic checkpoint,
or any additional files not listed above. The released files support
analysis of the stored estimates; they may not be sufficient to regenerate
all outputs from scratch.

See the repository's experiment scripts for the available processing
steps. The GSM8K dataset and base model remain subject to their respective
licenses and terms of use.
