---
title: "Mamba-3 Kernels (Equation-to-Code Map)"
date: 2026-05-12T14:30:00-07:00
draft: true
author: "Kartik"
tags: ["mamba", "triton", "kernels"]
categories: ["misc"]
description: "Full equation-by-equation mapping of the Mamba-3 EMA kernels to Triton code variables."
---

This version keeps all the core equations from my original Mamba-3 post, and maps each one to concrete variables in `new-ema-kernels`.

Repo/branch used: <https://github.com/kartiksrinivas007/triton-kernels/tree/new-ema-kernels>

## Notation map used below

- `X_c` -> `x_block`
- `O_c` -> `do_block` / `dO_reloaded` in backward (adjoint context)
- `F_(c-1)` (chunk start/final previous) -> `ssm_states_block` loaded from `SSM_States`
- `\bar{F_c}` carry -> `d_ssm_states_acc`
- `A_cs` -> `da_cs` (and `exp_da_cs = exp2(da_cs)`)
- `A_cs_rev` -> `da_cs_rev = da_cs_chunk_sum - da_cs` (and `exp_da_cs_rev`)
- `A_chunk_decay(c)` -> `exp2(da_cs_chunk_sum)`

Code pointers:
- `kernels/forward/ema_ssd_fwd.py` (`ema_fwd_kernel`)
- `kernels/backward/ema_ssd_bwd.py` (`ema_ssd_bwd_kernel_dpx`)
- `kernels/backward/ema_ssd_combined.py` (`compute_dpx`, `_EmaFunction`)

## Main output equation

$$
\begin{equation}
    (L \odot CB^T) X_c + (C \times F_{c - 1}^T) \odot A_{cs} = O_c
\end{equation}
$$

Equation -> code mapping:

- First term `(L \odot CB^T)X_c` is implemented as the masked/decayed intra-chunk matmul path.
  - forward: `s_block = exp2(segsum_triton(da_chunk, CHUNK_SIZE))`, then `acc_o += dot(s_block, x_block)`
  - backward adjoint contribution appears via `dAinv = dot(x_block, trans(do_block))` with causal decay mask.

- Second term `(C x F_(c-1)^T) \odot A_cs` is the state-to-output path.
  - backward: `sum(trans(ssm_states_block) * dO_reloaded, axis=1) * exp_da_cs`
  - this contributes into `dM_rev_vector`.

## State recursion equations

$$
\begin{equation}
    F_{c} = S_{c} + F_{c - 1} \times A_{chunk-decay}(c)
\end{equation}
$$

$$
\begin{equation}
    S_{c} = (X_c \odot A_{cs-rev})^T B
\end{equation}
$$

Equation -> code mapping:

- `A_chunk_decay(c)` maps to `exp2(da_cs_chunk_sum)`.
- forward recurrence update:

```python
acc_states *= exp2(dacs_last)
acc_states += dot(trans(exp2(dacsrev_chunk)[:, None]), x_block)
```

- `S_c` term is exactly the second addend above, with `A_cs_rev` represented by `dacsrev_chunk`.
- backward uses the same decomposition via `dM_vector` (reverse-cumsum path) and `dM_scalar` (chunk-decay scalar path).

## Gradient of X equation

$$
\bar{X_c} = (L^T \odot C^T B) \bar{O} +  B \bar{S_c}^T \odot A_{cs-rev}
$$

Equation -> code mapping:

- First term `(L^T \odot C^T B)\bar{O}`:

```python
acc_dx = dot(p_t_block, do_block)
```

where `p_t_block` is the causal decayed mask (`L^T`-style weighting).

- Second term `B \bar{S_c}^T \odot A_cs_rev`:

```python
acc_dx += trans(d_ssm_states_acc) * exp_da_cs_rev[:, None]
```

Since `\bar{S_c} = \bar{F_c}`, the carry variable is `d_ssm_states_acc`.

## Obtaining gradients of F_(c-1)

From `O_c` path:

$$
\bar{F_{c - 1}} = (\bar{O} \odot A_{cs})^T \times C
$$

From recurrence coupling path:

$$
\bar{F_{c - 1}} = \bar{F_c} \times A_{chunk-decay}(c)
$$

Equation -> code mapping:

- First path (from output in current chunk):

```python
dO_reloaded *= exp_da_cs[:, None]
state_from_output = sum(trans(dO_reloaded), axis=1, keep_dims=True)
```

- Second path (carry decay across chunks):

```python
state_from_carry = exp2(da_cs_chunk_sum) * d_ssm_states_acc
```

- Combined carry update:

```python
d_ssm_states_acc = state_from_carry + state_from_output
```

This is exactly the reverse-time accumulator logic.

## Tracking gradients of A: all four points

### Point 1: gradients through L

$$
\overline{L} = (\bar{O} X^T) \odot CB^T
$$

Equation -> code mapping:

- kernel computes `dot(x_block, trans(do_block))`, then applies causal-decay mask.
- this is `dAinv` and starts `dM_rev_vector`.

```python
dAinv = dot(x_block, trans(do_block))
dAinv *= causal_decay_mask
dM_rev_vector = sum(dAinv, axis=0) - sum(dAinv, axis=1)
```

### Point 2: gradients through A_cs

$$
\overline{A_{cs}} = \bar{O} \odot (C \times F_{c - 1}^T)
$$

Equation -> code mapping:

```python
dM_rev_vector += sum(trans(ssm_states_block) * dO_reloaded, axis=1) * exp_da_cs
```

This is your earlier `QS * dO`-style term, specialized to EMA simplifications.

### Point 3: gradients through A_cs_rev

$$
B \overline{S_c}^T \odot X_c = B \overline{F_c}^T \odot X_c = \overline{A_{cs-rev}}
$$

Equation -> code mapping:

```python
dM_vector = sum(trans(d_ssm_states_acc) * x_block_reloaded, axis=1) * exp_da_cs_rev
```

This is exactly the reverse-cumsum-side gradient vector.

### Point 4: gradients through A_chunk_decay

$$
\overline{A_{chunk-decay}}(c) = \sum \bar{F_c} \times F_{c - 1}
$$

Equation -> code mapping:

```python
dM_scalar = sum(SSM_States_reloaded * d_ssm_states_acc) * exp2(da_cs_chunk_sum)
```

`SSM_States` stores the chunk start state, which corresponds to `F_(c-1)` for chunk `c`.

## Final gradient accumulation identity in code

Kernel line:

```python
dM_rev_vector += (sum(dM_rev_vector) + dM_scalar) + cumsum(dM_vector - dM_rev_vector) - dM_vector
```

This merges:

1. reverse-cumsum conversion for `A_cs_rev`-side grads (`dM_vector` path),
2. chunk-decay scalar broadcast (`dM_scalar`),
3. forward-cumsum-side correction for `A_cs`-parameterization.

Your mapping intuition is right:

$$
\[a_2 + a_3 + a_4 , a_3 + a_4, a_4, 1\] \longrightarrow \[0, g_1, g_1 + g_2, g_1 + g_2 + g_3\]
$$

$$
[a_1 + a_2 + a_3 + a_4] \longrightarrow  [g, g, g, g]
$$

$$
\[a_1, a_1 + a_2, a_1 + a_2 + a_3, a_1 + a_2 + a_3 + a_4\] \longrightarrow \[g_1, g_1 + g_2, g_1 + g_2 + g_3, g_1 + g_2 + g_3 + g_4\]
$$

In code form this appears through the single fused line above for register/locality reasons.

## Wrapper glue (why these tensors appear in backward)

In `ema_combined`:

- forward stores `states`, `da_cs`, `da_cs_sum`
- states are reshaped to `ssm_states_shifted`
- backward calls `compute_dpx(x, da_cs, da_cs_sum, ssm_states, grad_out, ...)`

This is why the backward kernel can directly evaluate all equation terms without recomputing forward intermediates.

## Files cross-referenced

- `kernels/forward/ema_ssd_fwd.py`
- `kernels/backward/ema_ssd_bwd.py`
- `kernels/backward/ema_ssd_combined.py`
- `kernels/tests/ema_ssd_bwd/test_ema_dpx.py`
- `kernels/tests/ema_ssd_bwd/test_ema_combined_autograd.py`
