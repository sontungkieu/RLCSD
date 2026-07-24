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
| Tunix Sampler KV-cache decode | no | the current MaxText adapter has no `init_cache` and returns `logits, None` |
| End-to-end RLCSD | no | rollout, verifier, correct/wrong teacher contexts and the actor update are not yet connected |

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

## Matrix

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

All cases use sequence length 512 and the pinned dataset revision
`33d7de919af5b03257ff92c30303fddf9afdda4a`:

- Qwen3-1.7B: `deepmath_filtered_level5_7/train.parquet`
- Qwen3-4B: `deepmath_filtered_level6_8/train.parquet`
- Qwen3-8B: `deepmath_filtered_level7_10/train.parquet`

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

## Decode and end-to-end boundary

The installed MaxText `TunixMaxTextAdapter` currently accepts a `cache`
placeholder only for call-signature compatibility. It has no cache initializer,
does not update cache, and returns `None` as its second output. Record that
capability boundary without spending a TPU session:

```bash
python -m benchmarks.maxtext_tpu.benchmark \
  --mode decode-capability \
  --output outputs/maxtext_tpu/decode_capability.json
```

The next backend contribution is therefore a real cache-aware adapter (or a
pipeline-compatible MaxText decode path), followed by rollout/verifier/teacher
orchestration. Until those land, no output from this directory is an end-to-end
RLCSD training benchmark.
