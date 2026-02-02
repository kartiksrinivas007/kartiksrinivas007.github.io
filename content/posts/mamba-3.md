---
title: "Mamba-3 Kernels"
date: 2025-10-30T10:00:00-05:00
draft: true
author: "Kartik"
tags: ["mamba"]
categories: ["misc"]
description: "The Mamba-3 Kernels"
---


**Note:** This is a rough, work-in-progress post — things may change or be incomplete.



## Backward pass 

Main equations are

$$
\begin{equation}
    (L \odot CB^T) X_c + (C \times F_{c - 1}^T) \odot A_{cs} = O_c
\end{equation}
$$

Where F is states of shape `(head_dim, d_state)`

To get the gradients of X_c, we must also use other equations, since `F_c` depends on `X_c` which inturn affects of `O_(c + 1)`

$$
\begin{equation}
    F_{c} = S_{c} + F_{c - 1} \times A_{chunk-decay}(c)
\end{equation}
$$

With `F_0` as all zeroes and the chunk indexes are from `1, 2... num_chunks`

$$
\begin{equation}
    S_{c} = (X_c \odot A_{cs-rev})^T B
\end{equation}
$$

So the gradients to `X_c` have two parts, one through the first term, and the other through `S_c`.
The gradients are
$$
    \bar{X_c} = (L^T \odot C^T B) \bar{O} +  B \bar{S_c}^T \odot A_{cs-rev} 
$$

Note that \\(\bar{S_c}  = \bar{F_{c}}\\), so we need a variable that keeps track of the gradient of the final state chunks, this is `d_ssm_states_acc`.



### Obtaining gradients of `F_c`

The first part of the gradient of `F_{c - 1}` is through `O_c`

$$
\bar{F_{c - 1}} = (\bar{O} \odot A_{cs})^T \times C
$$

The second part of the gradient is the indirect `effect` of `F_(c - 1)` on `F_c` from equation (2)

$$
\bar{F_{c - 1}} = \bar{F_c} \times A_{chunk-decay}(c)
$$

We add both of these gradient flows to get the full adjoint of `F_c` = `S_c`'s adjoint as well.

At any iteration, we maintain `d_ssm_states_acc` which is `F_c` since the kernel goes in reverse, at the end of the iteration, we update this accumulator to hold adjoint of `F_(c - 1)`

### Tracking the Gradients of A

The gradients of A are tricky, A comes in multiple areas, namely in

1. In the production of the segsum mask, `L`
2. In `A_cs`
3.  In `A_cs_rev`
4. In `A_chunk_decay`

Let us go point by point 

#### Point 1

We need the segsum mask gradients in terms of the gradients of `A_cs`

$$
\overline{L} = (\bar{O} X^T) \odot CB^T
$$


#### Point 2

From Equation (1) we get

$$
\overline{A_{cs}} = \bar{O} \odot (C \times F_{c - 1}^T)
$$

This is line 564
```
dM_rev_vector += tl.sum(QS * dO_reloaded, axis=1) * exp_da_cs  # (CHUNK_SIZE,)
```

#### Point 3
From Equation (3) we get

$$
B \overline{S_c}^T \odot X_c = B \overline{F_c}^T \odot X_c = \overline{A_{cs-rev}} 
$$

This is `dM_vector` in the code, note that is this the gradient with respect to reverse cumsums

#### Point 4

$$
\overline{A_{chunk-decay}}(c) = \sum \bar{F_c} \times F_{c - 1}
$$

Note that in the kernel `SSM_states` holds the "beginning" state for each chunk, so `F_(c - 1)`, which is the final state of chunk c - 1, is actually the starting state of the `c`'th chunk.
This is `dM_scalar` in the code.


### The Final Gradient accumulation

In this step we decode this line 

```
   dM_rev_vector += (tl.sum(dM_rev_vector) + dM_scalar) + tl.cumsum(dM_vector - dM_rev_vector) - dM_vector
```

We can do it in 2 steps after noting `cumsum(a - b) = cumsum(a) - cumsum(b)`

First part is to compute pure A gradients from `dM_vector` which is gradients with respect to `dA_cs_rev`. Let the gradients of `dA_cs_rev = dM_vector = g_i`, then the pure gradients are this map

$$
\[a_2 + a_3 + a_4 , a_3 + a_4, a_4, 1\] \longrightarrow \[0, g_1, g_1 + g_2, g_1 + g_2 + g_3\]
$$

This is precisely `tl.cumsum(dM_vector) - dM_vector`

Second part is to distribute the gradients of chunk decay `dM_scalar = g`

$$
[a_1 + a_2 + a_3 + a_4] \longrightarrow  \[g, g, g, g\]
$$

The final step is to make gradients of `dA_cs` 


$$
\[a_1, a_1 + a_2, a_1 + a_2 + a_3, a_1 + a_2 + a_3 + a_4\] \longrightarrow \[g_1, g_1 + g_2, g_1 + g_2 + g_3, g_1 + g_2 + g_3 + g_4\]
$$

This is precisely the map `dM_rev_vector + tl.sum(dM_rev_vector) - tl.cumsum(dM_rev_vector)`













