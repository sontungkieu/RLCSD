# RLCSD on Tunix/MaxText TPU v5e-8

This directory turns the earlier forward-only experiments into an
evidence-gated integration path. It keeps different claims separate:

| Phase | Implemented here | What a passing result proves |
|---|---:|---|
| Matrix/config validation | yes | model, RLCSD config, dataset, batch and PP/TP layout are internally consistent |
| HF to MaxText checkpoint conversion | yes | an Orbax checkpoint was produced from one resolved HF revision |
| HF versus MaxText logits parity | yes | converted weights preserve one real prompt's next-token logits within declared tolerances |
| Full-sequence forward | yes | inference-style forward throughput on real RLCSD prompts |
| Core train step | yes | forward, causal-LM loss, backward and AdamW update fit and run |
| RLCSD policy-loss kernel | yes | the K-marginal, sign-preserving two-path objective is available in JAX |
| MaxText KV-cache rollout | implemented, TPU evidence pending | `OfflineEngine` produces 64×8 on-policy samples and token log-probabilities |
| Original-config RLCSD core update | implemented, TPU evidence pending | verifier, self-excluded teacher contexts, K-marginal loss, backward and AdamW are connected |
| Full 30-epoch RLCSD reproduction | implementation complete, terminal evidence pending | resumable session shards retain the dataset, validation and save/eval contracts |

Accordingly, neither a forward result nor a core causal-LM train step may be
reported as “RLCSD training throughput.” `audit.py` rejects random weights and
incomplete evidence by default.

## Reproducible dependency

The TPU requirements pin the exact Tunix fork commit that introduces
`MaxTextPipelineConfig`:

```bash
python -m pip install -r benchmarks/maxtext_tpu/requirements-tpu.txt
```

Do not commit `HF_TOKEN`, `kaggle.json`, converted checkpoints, downloaded
datasets, or benchmark outputs.

## Canary matrix

Generate and validate all 18 cases (3 models × 3 batches × 2 layouts):

```bash
python -m benchmarks.maxtext_tpu.matrix \
  --output outputs/maxtext_tpu/matrix.json
```

The layouts both use eight TPU devices:

- `PP2xTP4`: four microbatches for batch 12/16/64.
- `PP4xTP2`: four microbatches for batch 12, eight for batch 16/64.

The batch-12 PP4 result therefore is not a strict microbatch-count-controlled
comparison with batch 16/64. Batch 64 is attempted only as a measured case; the
matrix does not imply that backward, AdamW state, teacher passes, or KV cache
will fit.

All 18 matrix cases use sequence length 512. They are infrastructure canaries,
not the original RLCSD training configuration. `audit.py` will reject a
sequence-512 case if it is relabelled as original RLCSD evidence. The matrix
uses the pinned dataset revision
`33d7de919af5b03257ff92c30303fddf9afdda4a`:

- Qwen3-1.7B: `deepmath_filtered_level5_7/train.parquet`
- Qwen3-4B: `deepmath_filtered_level6_8/train.parquet`
- Qwen3-8B: `deepmath_filtered_level7_10/train.parquet`

## Original Qwen3-1.7B RLCSD contract

The original profile is derived from
`configs/math_deepmath/qwen3_1_7b_rlcsd.yaml` plus the effective dimensions in
`scripts/_run_verl.sh`. Validate and serialize it before staging a run:

```bash
python -m benchmarks.maxtext_tpu.rlcsd_contract \
  --output outputs/maxtext_tpu/original_rlcsd/contract.json
```

The fail-closed contract includes 64 prompts per rollout, 8 samples per prompt
(512 samples), PPO mini-batch 16, micro-batch cap 4, prompt/completion lengths
2048/16384, validation/model budget 40960, actor token budget 40960, snapshot
teacher sync every 10 updates, and all optimization/RLCSD hyperparameters. The
TPU execution layout is TP8 so rollout and training can use all eight v5e
devices sequentially. This changes the hardware sharding from the reference
8-GPU DP layout but preserves the batch, token, sampling, teacher and loss
semantics; it must be reported as a TPU port rather than hardware-identical
reproduction.

Convert and parity-check a checkpoint with the original TP8 structure:

```bash
python -m benchmarks.maxtext_tpu.checkpoint \
  --case qwen3_1_7b-original-rlcsd-tp8 \
  --hf-revision main \
  --output-dir checkpoints/maxtext/qwen3_1_7b_original_tp8

python -m benchmarks.maxtext_tpu.parity \
  --case qwen3_1_7b-original-rlcsd-tp8 \
  --checkpoint-items checkpoints/maxtext/qwen3_1_7b_original_tp8/0/items \
  --output outputs/maxtext_tpu/original_rlcsd/parity.json
```

Generate one complete original rollout batch with MaxText's cache-aware
`OfflineEngine`:

```bash
python -m benchmarks.maxtext_tpu.rlcsd_rollout \
  --checkpoint-items checkpoints/maxtext/qwen3_1_7b_original_tp8/0/items \
  --output-dir outputs/maxtext_tpu/original_rlcsd
```

The entrypoint enables MaxText's `DECOUPLE_GCLOUD=TRUE` mode before importing
`OfflineEngine`. This avoids an undeclared optional Jetstream serving
dependency; RLCSD still runs the real MaxText model, KV-cache prefill and
autoregressive decode, and supplies its Hugging Face tokenizer and EOS ids
directly.

Then run every valid PPO mini-batch. Do not set `--max-updates` for auditable
evidence; that option is diagnostic only:

```bash
python -m benchmarks.maxtext_tpu.rlcsd_actor \
  --rollouts outputs/maxtext_tpu/original_rlcsd/rollouts.jsonl \
  --rollout-manifest outputs/maxtext_tpu/original_rlcsd/rollout_manifest.json \
  --checkpoint-items checkpoints/maxtext/qwen3_1_7b_original_tp8/0/items \
  --output outputs/maxtext_tpu/original_rlcsd/actor_update.json

python -m benchmarks.maxtext_tpu.audit \
  outputs/maxtext_tpu/original_rlcsd/rollout_manifest.json \
  outputs/maxtext_tpu/original_rlcsd/actor_update.json \
  --output outputs/maxtext_tpu/original_rlcsd/audit.json
```

The actor precomputes teacher log-probabilities in snapshot windows of ten
updates, filters to reference-valid mixed-outcome samples, uses the reference
teacher-aware Karmarkar-Karp workload partitioner under the 40,960-token
budget, applies the reference dynamic-batch loss scaling, token-level rollout
IS capped at 2, and performs clipped AdamW with the original warmup.

Run a resumable shard of the complete original contract with:

```bash
python -m benchmarks.maxtext_tpu.rlcsd_train \
  --checkpoint-items checkpoints/maxtext/qwen3_1_7b_original_tp8/0/items \
  --cache-dir data/hf \
  --output-dir outputs/maxtext_tpu/original_rlcsd/session_0001
```

The trainer first executes `val_before_train` with 12 sampled answers for
AMC23, AIME24 and AIME25. It then consumes the deterministic 60,000-row prefix
with `drop_last=True` (937 rollout batches per epoch), keeps AdamW and snapshot
teacher state across batches, evaluates and saves every 50 rollout steps, and
writes a full Orbax model/optimizer/teacher checkpoint at each session
boundary. Resume a later session with the previous artifact's checkpoint root:

```bash
python -m benchmarks.maxtext_tpu.rlcsd_train \
  --checkpoint-items checkpoints/maxtext/qwen3_1_7b_original_tp8/0/items \
  --resume-checkpoint-dir /path/to/previous/checkpoints \
  --cache-dir data/hf \
  --output-dir outputs/maxtext_tpu/original_rlcsd/session_0002
```

`audit.py` accepts an incomplete output only as
`phase=rlcsd_training_shard`. It accepts `phase=rlcsd_end_to_end` only after
all 28,110 drop-last rollout steps, 30 epochs, pre-train and every scheduled
validation, and every 50-step checkpoint ledger entry are complete.

## 1. Convert a real checkpoint

Run conversion on a CPU host with enough RAM and disk:

```bash
python -m benchmarks.maxtext_tpu.checkpoint \
  --case qwen3_1_7b-bs12-pp2xtp4 \
  --hf-revision main \
  --output-dir checkpoints/maxtext/qwen3_1_7b
```

The wrapper resolves `main` to an immutable HF commit before conversion and
writes `checkpoint_manifest.json`. It passes authentication only through the
environment and never serializes it. One converted checkpoint can be reused
for every batch/layout case of the same model; each benchmark still validates
the model ID and immutable HF revision from the manifest.

## 2. Gate the conversion with logits parity

On TPU v5e-8:

```bash
python -m benchmarks.maxtext_tpu.parity \
  --case qwen3_1_7b-bs12-pp2xtp4 \
  --checkpoint-items checkpoints/maxtext/qwen3_1_7b/0/items \
  --output outputs/maxtext_tpu/qwen3_1_7b/parity.json
```

Parity compares the same real dataset prompt and resolved HF revision. The
default absolute-error, cosine and top-k thresholds are recorded in the result;
changing them changes the claim and must remain visible.

## 3. Measure forward and a core train step

```bash
python -m benchmarks.maxtext_tpu.benchmark \
  --mode forward \
  --case qwen3_1_7b-bs12-pp2xtp4 \
  --checkpoint-items checkpoints/maxtext/qwen3_1_7b/0/items \
  --output outputs/maxtext_tpu/qwen3_1_7b/forward.json

python -m benchmarks.maxtext_tpu.benchmark \
  --mode core-train-step \
  --case qwen3_1_7b-bs12-pp2xtp4 \
  --checkpoint-items checkpoints/maxtext/qwen3_1_7b/0/items \
  --output outputs/maxtext_tpu/qwen3_1_7b/core_train_step.json
```

Both commands require JAX to expose exactly eight TPU v5e devices. Compilation
and warmup are timed separately from repeated steps. Results include real/padded
token throughput, examples/s, process RSS, per-device memory stats when JAX
provides them, model/dataset provenance, and the exact measurement boundary.

`--allow-random-weights` exists only for infrastructure smoke tests. The audit
will reject such results as scientific checkpoint/training evidence.

## 4. Audit results

```bash
python -m benchmarks.maxtext_tpu.audit \
  outputs/maxtext_tpu/qwen3_1_7b/parity.json \
  outputs/maxtext_tpu/qwen3_1_7b/core_train_step.json \
  --output outputs/maxtext_tpu/qwen3_1_7b/audit.json
```

The command exits non-zero when any gate fails.

## Decode and full-training boundary

The installed MaxText `TunixMaxTextAdapter` remains a full-sequence training
adapter and has no cache initializer. Rollout therefore uses MaxText's separate
cache-aware `OfflineEngine`; it must not be described as Tunix Sampler decode.
The legacy adapter capability probe remains available:

```bash
python -m benchmarks.maxtext_tpu.benchmark \
  --mode decode-capability \
  --output outputs/maxtext_tpu/decode_capability.json
```

`rlcsd_actor.py` proves at most one original-config rollout/actor update batch
and deliberately emits `is_rlcsd_end_to_end=false`. `rlcsd_train.py` supplies
the full resumable loop, but implementation and a submitted/running shard are
not terminal evidence. The audit still requires all 30 epochs and every
scheduled validation/checkpoint before accepting an `rlcsd_end_to_end` claim.
