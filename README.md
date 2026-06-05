# RLCSD

Reinforcement Learning with Contrastive Self-Distillation — on-policy RL with K-marginal wrong-hint teacher contrast, built on [verl](https://github.com/volcengine/verl).

This repo contains the code and training configs for the paper experiments on math reasoning (DeepMath) and logic puzzles (Knights & Knaves) with Qwen3-1.7B / 4B / 8B and Olmo3-7B-Think.

## Repo layout

```
src/                       Trainer entrypoints, losses, data utilities
configs/
  math_deepmath/           Math configs (one per model × algorithm)
  logic_kk/                Logic configs
scripts/
  _run_verl.sh             Launcher: reads a YAML and runs verl PPO
  math_deepmath/run_*.sh   Per-config shims
  logic_kk/run_*.sh
  download_data.py         Pull train/eval parquets from HuggingFace
third_party/verl/          Vendored verl with RLCSD policy losses registered
requirements.txt
```

## Install

```bash
# Recommended: a fresh Python 3.10–3.12 env
pip install -r requirements.txt
```

`third_party/verl/` is added to `PYTHONPATH` automatically by `_run_verl.sh`.
flash-attn and vLLM must be built against your CUDA version — see their docs if
the wheels above don't match your environment.

## Data

The training and eval parquets live at
[Leyiii/RLCSD](https://huggingface.co/datasets/Leyiii/RLCSD) (currently private).
Pull everything in one shot:

```bash
# Authenticate first if the dataset is still private:
#   huggingface-cli login
python scripts/download_data.py --all
```

This writes to `data/verl/<dataset>/{train,val}.parquet`. The launcher resolves
paths under that root.

| Dataset                          | Used by                  |
|----------------------------------|--------------------------|
| `deepmath_filtered_level5_7`     | Qwen3-1.7B (math)        |
| `deepmath_filtered_level6_8`     | Qwen3-4B (math)          |
| `deepmath_filtered_level7_10`    | Qwen3-8B + Olmo (math)   |
| `amc23+aime24+aime25`            | math eval                |
| `kk_4to8`                        | logic train              |
| `kk_4to8_test+kk_9+kk_10+kk_11`  | logic eval               |

## Run

```bash
# RLCSD on Qwen3-4B, math
bash scripts/math_deepmath/run_qwen3_4b_rlcsd.sh

# SDPO baseline on Olmo3-7B-Think, logic
bash scripts/logic_kk/run_olmo3_7b_think_sdpo.sh
```

Each shim is a one-liner that forwards a config to `scripts/_run_verl.sh`. To
override individual hyperparameters, append Hydra-style overrides:

```bash
bash scripts/math_deepmath/run_qwen3_4b_rlcsd.sh learning_rate=2e-6 group_size=16
```

Optional environment overrides:

- `SWANLAB_API_KEY` — for swanlab logging (`use_swanlab: true` in configs)
- `HF_ENDPOINT` — for example, `https://hf-mirror.com` if you mirror HF
- `CUDA_HOME` — defaults to `/usr/local/cuda-12.6`

## Methods

Each YAML config selects a method via `method:`:

| Key          | Description                                                          |
|--------------|----------------------------------------------------------------------|
| `rlcsd`      | RLCSD: K-marginal teacher contrast (this paper)                      |
| `grpo`       | Vanilla GRPO baseline                                                |
| `opsd`       | On-policy self-distillation: forward KL / generalized JSD            |
| `sdpo`       | JSD distillation with importance-sampling correction (EMA teacher)   |
| `rlsd`       | GRPO with evidence-ratio modulated token-level advantages            |
| `srpo`       | Supervised ratio policy optimization                                 |
| `opsd_ectr`  | OPSD with token-level contrastive (e_ctr) masking                    |
| `rlsd_ectr`  | RLSD with teacher–teacher contrastive evidence ratio                 |

For RLCSD specifically:

- math: `learning_rate: 1e-6`
- logic: `learning_rate: 5e-6`
- `kl_loss_coef: 0`, `teacher_mode: snapshot`, K-multi controlled by `rlcsd_k_max`

## Citation

If you use this code or the released RLCSD method, please cite:

```bibtex
@article{rlcsd,
  title  = {RLCSD: Reinforcement Learning with Contrastive Self-Distillation},
  author = {...},
  year   = {2026}
}
```
