"""Bayesian surprise estimation for episodic memory triggers."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def bayesian_surprise(logits: jnp.ndarray, prior: jnp.ndarray) -> jnp.ndarray:
    """Computes Bayesian surprise as KL divergence from prior to posterior."""
    posterior = jax.nn.softmax(logits, axis=-1)
    prior = prior / jnp.sum(prior, axis=-1, keepdims=True)
    surprise = jnp.sum(posterior * (jnp.log(posterior + 1e-6) - jnp.log(prior + 1e-6)), axis=-1)
    return surprise
