---
title: "Mamba Notes"
date: 2025-10-30T10:00:00-05:00
draft: false
author: "Kartik"
tags: ["random","thoughts","coffee"]
categories: ["misc"]
description: "Mamba - Notes"
---

**Note:** This is a rough, work-in-progress post — things may change or be incomplete.

This post describes the triton implementation of the backward pass of the [Mamba-2](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/) Chunking sequence layer.

We use adjoint notation, i.e \\(\bar{A} \\) means  \\(\frac{\partial L} {\partial {A}} \\).The only important result to keep in mind here is how adjoints flow over matmuls, and that broadcasting calls for sum over the partial gradients over the dimension that was broadcasted on in the first place

$$ AX = S \implies \bar{S} X^T = \bar{A} \ \ \ \rightarrow  \ \  A^T \bar{S} = \bar{X}$$

### Kernel 1 `chunk_scan_bwd_dstates`

`chunk_scan_fwd` kernel, matmul over `d_state`

$$(C \odot A') \times F + \ldots = O$$


`chunk_scan_bwd_dstates` kernel matmul over `chunk_size`

$$(A' \odot C^T) \times \bar{O} = \bar{F}$$

With \\(A'C^T \\) shape = `(d_state, Q)` and  \\(\bar{O} \\) shape = `(Q, h_dim)` and  \\(\bar{F} \\) shape = `(d_st, h_dim)`
This is computed over a grid of shape `(batch, chunks, num_heads, ...)` and blocked over `(head_dim, d_state)`, with [tiled matrix multiplication](https://penny-xu.github.io/blog/tiled-matrix-multiplication) over `chunk_size`


Further more, the same C and A' are used, so the gradients need to be summed over the number of heads if we are looking at those factors.

In the EMA case, we do not need to do the following 

1. We do not need to load a `C`
2. We need to load `A` -> Can we directly store gradients of `A_cs`?
3. We do need the `dstates`
4. We can also block over `token_dim` 



### Kernel 2 `_state_passing_bwd`

Forward pass


$$F_c = A\_{c - 1} F\_{c - 1} + s\_{c  - 1}$$

Backward pass

$$
\begin{equation*}
    \overline{s_{c - 1}} = \overline{F_{c}}   = \overline{F}_{c} + A\_{c-1} \overline{F}\_{c + 1}
\end{equation*}
$$

$$
\begin{equation*}
    \overline{A_{c - 1}}   = \langle F\_{c - 1} \overline{F}\_c \rangle
\end{equation*}
$$

Note that \\(A\_{c -1} = \exp(A_{chunk})\\) -- so one more step is needed (multiply again as \\(\partial e^x  = e^x\\))
The shape of each tensor is `(head_dim * d_state)`, but the A factor is simply a scalar.
Since A is **broadcasted** and the joint product dim is blocked with `block_size` we need to store the sum (or inner product) separately for each of the programs that are blocking and then add it up in the end, the grid is `(batch, ..)` 

For EMA 

1. We might need to also block over the batch dimension 
2. We might need to also store dA in separate pieces per program and add them later

### Kernel 3 `_chunk_state_bwd_db`


Forward pass 

$$
\begin{equation}
    (B^T \odot A) \times X = S
\end{equation}
$$

Here the shapes are B = `(d_state, Q)` and X = `(Q, head_dim)` and S = 
`(d_state, head_dim)`. In the forward pass this is a tiled matrix multiplication over the chunk_size dimension.

Backward Pass

$$
\begin{equation}
    (\bar{S} \times X^T) \odot A  = \bar{B}^T
\end{equation}
$$


$$
\begin{equation}
    (\bar{S} \times X^T) \odot B^T  = \bar{A}\_{d \times Q}
\end{equation}
$$
$$
\begin{equation}
    \bar{A}\_{Q} = \sum_d \bar{A}\_{d \times Q}
\end{equation}
$$

Note that since A is broadcasted, we need to add it over the `d-state` dimension to obtain the true adjoint of A.

For EMA

1. I need not compute the adjoint of B
2. In our case, the inner product would be over `(1 , token_dim) @ (token_dim, chunk_size)`. This differs from their design because the `head_dim` is actually smaller for them once they make multiple heads.
3. We do not need to iterate over multiple heads, we do need to do a tiled mm over `token_dim`, unlike them


### Kernel 4 `_chunk_scan_bwd_dC`

The complete equation for the forward of chunk_scan is

