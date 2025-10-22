"""Training loop optimized for TPU via pjit."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import struct
from flax.training import train_state
from jax.experimental import mesh_utils
from jax.experimental import pjit

from quietreasoning.config import QuietReasoningConfig
from quietreasoning.model import QuietReasoningModel
from quietreasoning.training.losses import compute_total_loss
from quietreasoning.training.optimizer import build_optimizer

Array = jnp.ndarray


@struct.dataclass
class QuietTrainState(train_state.TrainState):
    tokens_seen: Array


def create_train_state(rng: jax.Array, cfg: QuietReasoningConfig) -> QuietTrainState:
    model = QuietReasoningModel(cfg.model)

    dummy_input = jnp.zeros((1, cfg.model.context), dtype=jnp.int32)
    variables = model.init({"params": rng}, dummy_input, deterministic=True)
    params = variables["params"]

    optimizer = build_optimizer(cfg.training.optimizer)

    return QuietTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        tokens_seen=jnp.array(0.0),
    )


def make_mesh(num_devices: int) -> jax.sharding.Mesh:
    devices = mesh_utils.create_device_mesh((num_devices,))
    return jax.sharding.Mesh(devices, ("data",))


def build_train_step(cfg: QuietReasoningConfig):
    @partial(
        pjit.pjit,
        in_shardings=(None, None, None, None),
        out_shardings=None,
        donate_argnums=(0,),
    )
    def train_step(
        state: QuietTrainState,
        batch: Dict[str, Array],
        step_tokens: float,
        feature_gates: Dict[str, Array],
    ) -> Tuple[QuietTrainState, Dict[str, Any]]:
        rng = jax.random.fold_in(jax.random.PRNGKey(0), state.step)

        def loss_fn(params):
            outputs = state.apply_fn(
                {"params": params},
                batch["input_ids"],
                batch.get("attention_mask"),
                stage_features=feature_gates,
                rngs={"dropout": rng},
            )
            loss, logs = compute_total_loss(outputs, batch, cfg)
            return loss, (logs, outputs)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logs, outputs)), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(tokens_seen=state.tokens_seen + step_tokens)
        logs = dict(logs)
        logs["loss"] = loss
        logs["workspace_steps"] = jnp.mean(outputs.workspace_steps)
        logs["ssm_gate"] = jnp.mean(outputs.router_decisions.ssm_gate)
        return state, logs

    return train_step
