# RLCSD on Talapas

This file is the repository-local source of truth for the RLCSD Talapas smoke
workflow. It does not contain credentials.

## Paths

- Repository: `/gpfs/projects/ailab/tnguye11/sontungkieu/RLCSD`
- Peer repository: `/gpfs/projects/ailab/tnguye11/sontungkieu/Collaborative-MORL`
- Node-local environment: `/tmp/${USER}_rlcsd-cu13-sm80`
- Persistent environment archive: `.talapas/envs/rlcsd-cu13-sm80.tar.gz`
- Persistent FlashAttention wheel: `.talapas/wheelhouse/rlcsd-cu13-sm80/`
- Shared caches: `/scratch/ailab/tnguye11/rlcsd/`
- Slurm logs and training outputs: `outputs/talapas/`

PyTorch environments are built and imported from node-local `/tmp`. The setup
job archives the completed environment to GPFS, and the training job extracts
it back to the same node-local path. This avoids the observed Talapas failure
mode where CUDA PyTorch imports directly from a GPFS virtual environment stall.

## Runtime profile

- Slurm account: `ailab`
- Partition: `gpu`
- GRES: `gpu:nvidia_a100_80gb_pcie_3g.40gb:1`
- CUDA module: `cuda/13.0`
- Compiler module: `gcc/13.1.0`
- Python: `3.10`, managed by `uv`
- vLLM: `0.24.0`
- FlashAttention: `2.8.3.post1`, compiled for `sm_80`

## Submit

From the repository root on Talapas:

```bash
bash scripts/talapas/submit_w4_smoke.sh
```

The wrapper submits environment setup when its archive is absent, then submits
the three-epoch training smoke with an `afterok` dependency. Set
`FORCE_SETUP=1` to rebuild the environment. Resource settings can be overridden
with `SLURM_ACCOUNT`, `SLURM_PARTITION`, and `SLURM_GRES`.

Inspect jobs and logs with:

```bash
squeue -u "$USER"
tail -F outputs/talapas/slurm/setup_<job-id>.out
tail -F outputs/talapas/slurm/train_<job-id>.out
```

The setup job must emit `RLCSD_TALAPAS_ENV_OK`. The training job must emit
`RLCSD_TALAPAS_TRAIN_OK` and writes `metrics.jsonl`, `summary.json`,
`checkpoints.txt`, TensorBoard events, the copied config, and every
`global_step_*` checkpoint under its timestamped run directory.
