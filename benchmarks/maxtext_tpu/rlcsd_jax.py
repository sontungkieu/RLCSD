"""JAX implementation of the RLCSD policy-loss kernel.

This ports the tensor algebra from ``compute_policy_loss_rlcsd``. It does not
perform rollout sampling, verifier scoring, or teacher-context construction;
those remain separate end-to-end phases.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _masked_mean(values: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(values.dtype)
    return jnp.sum(values * mask) / jnp.maximum(
        jnp.sum(mask), jnp.asarray(1.0, values.dtype)
    )


def rlcsd_policy_loss(
    *,
    old_log_prob: jax.Array,
    log_prob: jax.Array,
    advantages: jax.Array,
    response_mask: jax.Array,
    teacher_correct_log_prob: jax.Array,
    teacher_wrong_multi_log_prob: jax.Array,
    teacher_wrong_multi_valid_mask: jax.Array,
    epsilon: float = 0.2,
    tau: float = 0.02,
    beta: float = 1.0,
    lam: float = 0.5,
    delta: float = 0.02,
    eta: float = 1.0,
    residual_clip_low: float = -2.0,
    residual_clip_high: float = 2.0,
    rollout_is_weights: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, Any]]:
    """Computes the two-path, K-marginal RLCSD PPO loss.

    Shapes are ``[B, T]`` except the wrong-hint log probabilities, which are
    ``[B, K, T]``, and their validity mask, which is ``[B, K]``.
    """
    if residual_clip_low > residual_clip_high:
        residual_clip_low, residual_clip_high = (
            residual_clip_high,
            residual_clip_low,
        )
    tau = max(float(tau), 1e-6)
    delta = max(float(delta), 0.0)
    eta = max(float(eta), 0.0)

    log_ratio = log_prob - old_log_prob
    ratio = jnp.exp(log_ratio)
    ratio_clamped = jnp.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)

    valid = teacher_wrong_multi_valid_mask.astype(bool)
    valid_count = jnp.maximum(
        jnp.sum(valid, axis=1),
        jnp.asarray(1, teacher_wrong_multi_valid_mask.dtype),
    )
    masked_wrong = jnp.where(
        valid[..., None],
        teacher_wrong_multi_log_prob,
        -jnp.inf,
    )
    log_marginal_wrong = (
        logsumexp(masked_wrong, axis=1)
        - jnp.log(valid_count.astype(log_prob.dtype))[:, None]
    )

    e_ctr = teacher_correct_log_prob - log_marginal_wrong
    residual = jnp.clip(
        beta * lam * jnp.tanh(e_ctr / tau),
        residual_clip_low,
        residual_clip_high,
    )
    selected = ((jnp.abs(residual) > delta) & response_mask.astype(bool)).astype(
        log_prob.dtype
    )
    residual_for_advantage = selected * residual
    raw_modulated_advantage = advantages + residual_for_advantage
    modulated_advantage = jnp.where(
        advantages > 0,
        jnp.maximum(raw_modulated_advantage, 0.0),
        jnp.where(
            advantages < 0,
            jnp.minimum(raw_modulated_advantage, 0.0),
            0.0,
        ),
    )
    # Teacher-derived modulation is a target, not a differentiable path.
    selected = jax.lax.stop_gradient(selected)
    modulated_advantage = jax.lax.stop_gradient(modulated_advantage)

    nonselected_mask = response_mask * (1.0 - selected)
    selected_mask = response_mask * selected
    nonselected_per_token = -jnp.minimum(
        ratio * advantages,
        ratio_clamped * advantages,
    )
    selected_per_token = -jnp.minimum(
        ratio * modulated_advantage,
        ratio_clamped * modulated_advantage,
    )
    if rollout_is_weights is not None:
        weights = jax.lax.stop_gradient(rollout_is_weights)
        nonselected_per_token *= weights
        selected_per_token *= weights

    nonselected_loss = _masked_mean(nonselected_per_token, nonselected_mask)
    selected_loss = _masked_mean(selected_per_token, selected_mask)
    loss = nonselected_loss + eta * selected_loss
    metrics = {
        "loss": loss,
        "ppo_kl": _masked_mean(-log_ratio, response_mask),
        "clip_fraction": _masked_mean(
            (jnp.abs(ratio - 1.0) > epsilon).astype(log_prob.dtype),
            response_mask,
        ),
        "selected_token_count": jnp.sum(selected_mask),
        "response_token_count": jnp.sum(response_mask),
        "k_valid_mean": jnp.mean(valid_count.astype(jnp.float32)),
        "e_ctr_abs_mean": _masked_mean(jnp.abs(e_ctr), response_mask),
    }
    return loss, metrics
