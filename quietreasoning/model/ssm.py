"""Selective state-space layer with workspace gating."""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

Array = jnp.ndarray


def _stable_softplus(x: Array) -> Array:
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.0)


class SelectiveStateSpace(nn.Module):
    """Mamba-style selective scan with light-weight parameterization."""

    d_model: int
    state_dim: int
    conv_kernel: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        batch, length, _ = x.shape

        if self.conv_kernel > 1:
            pad = ((0, 0), (self.conv_kernel - 1, 0), (0, 0))
            conv_input = jnp.pad(x, pad)
        else:
            conv_input = x
        conv = nn.Conv(
            features=self.d_model,
            kernel_size=(self.conv_kernel,),
            padding="VALID",
            feature_group_count=self.d_model,
            dtype=self.dtype,
            name="depthwise_conv",
        )(conv_input)

        proj = nn.Dense(
            features=2 * self.state_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="input_proj",
        )(conv)
        B_t, C_t = jnp.split(proj, 2, axis=-1)
        B_t = nn.softplus(B_t).astype(self.dtype)
        C_t = jnp.tanh(C_t).astype(self.dtype)

        u_proj = nn.Dense(
            self.state_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="state_proj",
        )(x)

        A_log = self.param(
            "A_log",
            nn.initializers.normal(stddev=0.02),
            (self.state_dim,),
        )
        dt_log = self.param(
            "dt_log",
            nn.initializers.normal(stddev=0.02),
            (self.state_dim,),
        )
        A = -nn.softplus(A_log).astype(self.dtype)
        dt = nn.softplus(dt_log).astype(self.dtype)

        def scan_step(carry: Array, inputs: Tuple[Array, Array, Array]) -> Tuple[Array, Array]:
            state = carry
            u_t, b_t, c_t = inputs
            one = jnp.asarray(1.0, dtype=self.dtype)
            state = (one + dt * A) * state + dt * b_t * u_t
            y_t = c_t * state
            return state, y_t

        init_state = jnp.zeros((batch, self.state_dim), dtype=self.dtype)
        scan_inputs = tuple(jnp.swapaxes(arr, 0, 1) for arr in (u_proj, B_t, C_t))

        _, outputs = jax.lax.scan(
            scan_step,
            init_state,
            xs=scan_inputs,
            unroll=4,
        )

        outputs = jnp.swapaxes(outputs, 0, 1)

        outputs = nn.Dense(
            self.d_model,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="output_proj",
        )(outputs)
        return outputs


class WorkspaceGatedSSM(nn.Module):
    """Wraps SSM with workspace-conditioned gating."""

    d_model: int
    state_dim: int
    conv_kernel: int = 4
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: Array,
        workspace_summary: Optional[Array],
        deterministic: bool = True,
        global_gate: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        gate_input = workspace_summary
        if gate_input is None:
            gate_input = jnp.mean(x, axis=1)
        gate_logits = nn.Dense(
            1,
            dtype=self.dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            name="gate_dense",
        )(gate_input)
        gate_scalar = jax.nn.sigmoid(gate_logits)
        selector = jnp.squeeze(gate_scalar, axis=-1)
        if global_gate is not None:
            global_gate = jnp.asarray(global_gate, dtype=self.dtype)
            if global_gate.ndim == 0:
                global_gate = jnp.broadcast_to(global_gate, selector.shape)
            elif global_gate.shape[0] != selector.shape[0]:
                global_gate = jnp.broadcast_to(global_gate, selector.shape)
            selector = selector * global_gate

        ssm_out = SelectiveStateSpace(
            d_model=self.d_model,
            state_dim=self.state_dim,
            conv_kernel=self.conv_kernel,
            dtype=self.dtype,
        )(x)

        selector_expanded = jnp.expand_dims(selector, axis=(1, 2))
        ssm_out = selector_expanded * ssm_out
        residual_adjust = -selector_expanded * x
        total = ssm_out + residual_adjust
        return total, selector
