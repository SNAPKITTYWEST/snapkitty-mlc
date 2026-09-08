"""
FlashAttention Golden Model — Software Reference Implementation
Implements Dao et al. FlashAttention (Algorithm 1) with Tiling and Online Softmax

Demonstrates:
  1. Block-wise (tiled) computation of attention
  2. Online softmax with running max (m_i) and sum (l_i) scalars
  3. Numerical validation against standard attention

Reference:
  Dao, Dan, et al. "FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness." NeurIPS 2022.

Authors: Nova Parr, Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)
License: BSL-1.1 / AGPL-3.0 / MPL-2.0
"""

import numpy as np
from typing import Tuple, Dict
import math


class FlashAttentionGoldenModel:
    """
    FlashAttention software golden model with configurable block sizes (Br, Bc)
    and online softmax computation.

    Notation:
        Q  : (N, d) Query matrix
        K  : (N, d) Key matrix
        V  : (N, d) Value matrix
        Br : Block size for Q (query block)
        Bc : Block size for K, V (key/value block)
        d  : Embedding dimension
        m_i: Running maximum of logits (numerical stability)
        l_i: Running sum of exponentials (denominator)
    """

    def __init__(self, scale: float = None, eps: float = 1e-6):
        self.scale = scale
        self.eps   = eps

    # ── Numerically stable softmax ─────────────────────────────────────

    def _softmax_stable(self, logits: np.ndarray, scale: float = None) -> np.ndarray:
        if scale is not None:
            logits = logits * scale
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp     = np.exp(shifted)
        return exp / (np.sum(exp, axis=-1, keepdims=True) + self.eps)

    # ── Reference: standard O(N²) attention ───────────────────────────

    def attention_naive(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        scale: float = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Standard attention (reference / ground truth).
        Output = softmax(Q @ K^T / sqrt(d)) @ V
        """
        if scale is None:
            scale = 1.0 / math.sqrt(Q.shape[-1])

        scores  = Q @ K.T * scale
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(shifted)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        output  = weights @ V

        return output, {'attn_weights': weights, 'scores': scores}

    # ── FlashAttention Algorithm 1 ─────────────────────────────────────

    def attention_flash(
        self,
        Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        Br: int = 32, Bc: int = 32,
        scale: float = None, verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        FlashAttention Algorithm 1: tiled computation with online softmax.

        For each Q-block (size Br), iterate over all K/V-blocks (size Bc):
          · Compute block scores S_ij = Q_block @ K_block^T * scale
          · Online softmax: update running max m_i and sum l_i
          · Accumulate output O_i with appropriate rescaling
          · Final normalisation: O_i /= l_i
        """
        if scale is None:
            scale = 1.0 / math.sqrt(Q.shape[-1])

        N, d           = Q.shape
        num_q_blocks   = math.ceil(N / Br)
        num_kv_blocks  = math.ceil(N / Bc)
        O              = np.zeros_like(Q)
        m_history, l_history = [], []

        if verbose:
            print(f"FlashAttention  Q/K/V=({N},{d})  Br={Br}  Bc={Bc}  scale={scale:.6f}")

        for i in range(num_q_blocks):
            q0, q1   = i * Br, min((i + 1) * Br, N)
            Q_blk    = Q[q0:q1]                           # (br, d)
            br       = q1 - q0
            m_i      = np.full(br, -np.inf)
            l_i      = np.zeros(br)
            O_i      = np.zeros((br, d))

            for j in range(num_kv_blocks):
                k0, k1  = j * Bc, min((j + 1) * Bc, N)
                K_blk   = K[k0:k1]                        # (bc, d)
                V_blk   = V[k0:k1]                        # (bc, d)

                S_ij    = Q_blk @ K_blk.T * scale         # (br, bc)

                # Online softmax update
                m_ij    = np.max(S_ij, axis=1)            # (br,)
                m_new   = np.maximum(m_i, m_ij)
                P_ij    = np.exp(S_ij - m_new[:, None])   # (br, bc)
                l_ij    = np.sum(P_ij, axis=1)            # (br,)
                alpha   = np.exp(m_i - m_new)             # (br,)
                l_new   = alpha * l_i + l_ij
                O_i     = alpha[:, None] * O_i + P_ij @ V_blk
                m_i, l_i = m_new, l_new

            O[q0:q1] = O_i / l_i[:, None]
            m_history.append(m_i.copy())
            l_history.append(l_i.copy())

        return O, {
            'algorithm'  : 'FlashAttention Algorithm 1',
            'block_sizes': {'Br': Br, 'Bc': Bc},
            'm_history'  : m_history,
            'l_history'  : l_history,
        }

    # ── Validation ─────────────────────────────────────────────────────

    def validate(
        self,
        Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        Br: int = 32, Bc: int = 32,
        tolerance: float = 1e-4, verbose: bool = True
    ) -> Dict:
        """
        Compare FlashAttention against naive attention and report error metrics.
        """
        scale  = 1.0 / math.sqrt(Q.shape[-1])
        O_ref, _  = self.attention_naive(Q, K, V, scale)
        O_flash, aux = self.attention_flash(Q, K, V, Br, Bc, scale)

        abs_err  = np.abs(O_ref - O_flash)
        rel_err  = np.abs(abs_err / (np.abs(O_ref) + self.eps))
        passed   = np.max(rel_err) < tolerance

        if verbose:
            print(f"\n{'='*60}")
            print("VALIDATION RESULTS")
            print(f"{'='*60}")
            print(f"  Shape: {Q.shape}  Br={Br}  Bc={Bc}  tol={tolerance:.1e}")
            print(f"  Max abs error : {np.max(abs_err):.3e}")
            print(f"  Mean abs error: {np.mean(abs_err):.3e}")
            print(f"  Max rel error : {np.max(rel_err):.3e}")
            print(f"  Mean rel error: {np.mean(rel_err):.3e}")
            print(f"  Status        : {'PASS' if passed else 'FAIL'}")
            print(f"{'='*60}\n")

        return {
            'passed'        : passed,
            'max_abs_error' : np.max(abs_err),
            'max_rel_error' : np.max(rel_err),
            'mean_abs_error': np.mean(abs_err),
            'mean_rel_error': np.mean(rel_err),
            'O_naive'       : O_ref,
            'O_flash'       : O_flash,
        }


# ── Test suite ─────────────────────────────────────────────────────────

def run_golden_model_test() -> bool:
    print(f"\n{'='*60}")
    print("FlashAttention Golden Model Test Suite")
    print(f"{'='*60}\n")

    model = FlashAttentionGoldenModel()

    configs = [
        {'N':  4,  'd':  2, 'Br':  2, 'Bc':  2, 'tol': 1e-10, 'name': 'Sanity (N=4,d=2)'},
        {'N': 64,  'd': 32, 'Br': 16, 'Bc': 16, 'tol': 1e-5,  'name': 'Small  (N=64,d=32)'},
        {'N': 128, 'd': 64, 'Br': 32, 'Bc': 32, 'tol': 1e-5,  'name': 'Medium (N=128,d=64)'},
        {'N': 256, 'd':128, 'Br': 64, 'Bc': 64, 'tol': 1e-5,  'name': 'Large  (N=256,d=128)'},
    ]

    all_passed = True
    for cfg in configs:
        print(f"Test: {cfg['name']}")
        np.random.seed(42)
        Q = np.random.randn(cfg['N'], cfg['d']).astype(np.float64) * 0.1
        K = np.random.randn(cfg['N'], cfg['d']).astype(np.float64) * 0.1
        V = np.random.randn(cfg['N'], cfg['d']).astype(np.float64) * 0.1
        r = model.validate(Q, K, V, cfg['Br'], cfg['Bc'], cfg['tol'])
        all_passed = all_passed and r['passed']

    print(f"\n{'='*60}")
    print("ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED")
    print(f"{'='*60}\n")
    return all_passed


if __name__ == "__main__":
    run_golden_model_test()
