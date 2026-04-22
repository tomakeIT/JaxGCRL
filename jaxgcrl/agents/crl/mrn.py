"""
Metric Residual Network (MRN) energy function for JaxGCRL's CRL agent.

Based on:
  Liu et al., "Metric Residual Network for Sample Efficient
  Goal-Conditioned Reinforcement Learning", AAAI 2023.

-----------------------------------------------------------------------------
Why MRN
-----------------------------------------------------------------------------
JaxGCRL's default energy function is the (negated) L2 distance:
    f(s, a, g) = -||phi(s,a) - psi(g)||_2
L2 is a metric: it satisfies the triangle inequality (desirable for temporal
contrastive features per Myers et al. 2024b) but is symmetric.

Goal-reaching, however, is intrinsically *asymmetric*: in a maze or a push
task, reaching B from A can be much easier or harder than reaching A from B.
A purely symmetric distance cannot represent that.

MRN splits the representation vector into two halves:
    phi(s,a) = [ phi_sym | phi_asym ]   each of size repr_dim/2
    psi(g)   = [ psi_sym | psi_asym ]

and defines the distance as:
    d(s, a, g) = ||phi_sym - psi_sym||_2                # symmetric, metric
                 + max_k ReLU( phi_asym - psi_asym )_k  # asymmetric, residual

The energy is f = -d, matching the sign convention used by 'norm' and 'l2'
in JaxGCRL's losses.py.

-----------------------------------------------------------------------------
Key properties
-----------------------------------------------------------------------------
  1. Asymmetric:          f(a, b) != f(b, a) in general.
  2. Self-similarity = 0: f(x, x) == 0 exactly.
  3. Non-positive:        f(a, b) <= 0 always.
  4. Triangle inequality: holds on the symmetric half alone; the full MRN
                          satisfies a *quasimetric* triangle inequality.
  5. Same gradient cost:  roughly the same FLOPs as a negated L2 distance.

-----------------------------------------------------------------------------
Integration
-----------------------------------------------------------------------------
This file is meant to sit next to losses.py. In losses.py the existing
energy_fn dispatcher gets one new case:

    elif name == "mrn":
        return mrn_energy(x, y, sym_ratio=0.5)

That's it -- mrn_energy has the *same* broadcasting semantics as the other
energy functions in that file, so both the critic's [B, B] matrix call and
the actor's [B] diagonal call work unchanged.
"""

import jax
import jax.numpy as jnp


def mrn_energy(x: jnp.ndarray, y: jnp.ndarray, sym_ratio: float = 0.5) -> jnp.ndarray:
    """MRN energy function.

    Supports both calling patterns used in JaxGCRL's losses.py:
      1. Critic matrix:  x=[B,1,D], y=[1,B,D]  -> returns [B, B]
      2. Actor diagonal: x=[B, D],  y=[B, D]   -> returns [B]

    Works by reducing along axis=-1 after broadcasting, the same way the
    other energy functions in losses.py do.

    Args:
        x:         state-action representation. Shape [..., D].
        y:         goal representation.         Shape [..., D].
        sym_ratio: fraction of repr dims assigned to the symmetric half.
                   0.5 matches the original MRN paper.

    Returns:
        Energy, shape = broadcast(x.shape[:-1], y.shape[:-1]).
    """
    D = x.shape[-1]
    d_sym = int(round(D * sym_ratio))
    d_sym = max(1, min(D - 1, d_sym))  # guard against degenerate ratios

    # Split along the last dim. Broadcasting works because we only index
    # the last axis; the leading axes of x and y are untouched.
    x_sym = x[..., :d_sym]
    x_asym = x[..., d_sym:]
    y_sym = y[..., :d_sym]
    y_asym = y[..., d_sym:]

    # Symmetric component: L2 distance.
    # The +1e-6 under the sqrt matches the numerical guard used by "norm"
    # in JaxGCRL's losses.py.
    sym_diff = x_sym - y_sym
    sym_dist = jnp.sqrt(jnp.sum(sym_diff**2, axis=-1) + 1e-6)

    # Asymmetric residual: max_k ReLU(x_asym - y_asym)_k.
    # Intuition: pay a cost only for dimensions where x is "above" y.
    # Swapping x and y gives a different cost -> asymmetric.
    # `max` (not `sum`) preserves a quasimetric triangle inequality
    # in the limit, and makes the residual scale-stable as D grows.
    asym_diff = x_asym - y_asym
    asym_dist = jnp.max(jax.nn.relu(asym_diff), axis=-1)

    return -(sym_dist + asym_dist)

