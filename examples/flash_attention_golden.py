"""
FlashAttention Golden Model - Software Reference Implementation
Implements Dao et al. FlashAttention (Algorithm 1) with Tiling and Online Softmax

This model demonstrates:
1. Block-wise (tiled) computation of attention
2. Online softmax with running max (m_i) and sum (l_i) scalars
3. Numerical validation against standard attention

Key Reference: Dao, Dan, et al. "FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness" (2022)

Authors: Nova Parr, Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)
License: BSL-1.1 / AGPL-3.0 / MPL-2.0
"""

import numpy as np
from typing import Tuple, Dict, Callable
import math


class FlashAttentionGoldenModel:
    """
    FlashAttention software golden model with configurable block sizes (Br, Bc)
    and online softmax computation.

    Notation:
        - Q: (N, d) Query matrix
        - K: (N, d) Key matrix
        - V: (N, d) Value matrix
        - Br: Block size for Q (query block)
        - Bc: Block size for K, V (key/value block)
        - d: Embedding dimension
        - m_i: Running maximum of logits (for numerical stability)
        - l_i: Running sum of exponentials (denominator)
    """

    def __init__(self, scale: float = None, eps: float = 1e-6):
        """
        Args:
            scale: Scaling factor for Q*K^T (default: 1/sqrt(d))
            eps: Small constant for numerical stability
        """
        self.scale = scale
        self.eps = eps

    def _softmax_stable(self, logits: np.ndarray, scale: float = None) -> np.ndarray:
        """
        Numerically stable softmax computation.

        Args:
            logits: (batch, seq_len) raw attention scores
            scale: optional scaling factor

        Returns:
            Normalized attention weights (batch, seq_len)
        """
        if scale is not None:
            logits = logits * scale

        # Subtract max for numerical stability
        logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + self.eps)

    def attention_naive(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: float = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Standard attention (for validation reference).

        Output = softmax(Q @ K^T / sqrt(d)) @ V

        Args:
            Q, K, V: (N, d) matrices
            scale: scaling factor for Q @ K^T

        Returns:
            output: (N, d) attention output
            aux: metadata dict
        """
        if scale is None:
            scale = 1.0 / math.sqrt(Q.shape[-1])

        N = Q.shape[0]

        # Compute full attention matrix: (N, N)
        scores = Q @ K.T  # (N, N)
        scores = scores * scale

        # Numerically stable softmax
        scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
        attn_weights = np.exp(scores_shifted)
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

        # Apply to values: (N, d)
        output = attn_weights @ V

        return output, {
            'attn_weights': attn_weights,
            'scores': scores
        }

    def attention_flash(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        Br: int = 32,
        Bc: int = 32,
        scale: float = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """
        FlashAttention Algorithm 1: Tiled computation with online softmax.

        This implements the block-wise attention:
        - Partition Q into blocks of size Br
        - For each Q block, iterate over all K/V blocks (size Bc)
        - Maintain running max (m_i) and sum (l_i) for online softmax
        - Update output O incrementally

        Args:
            Q, K, V: (N, d) matrices
            Br: Query block size (rows of Q)
            Bc: Key/Value block size (rows of K, V)
            scale: Scaling factor for scores (default: 1/sqrt(d))
            verbose: Print debug info

        Returns:
            output: (N, d) attention output
            aux: metadata dict with online softmax stats
        """
        if scale is None:
            scale = 1.0 / math.sqrt(Q.shape[-1])

        N, d = Q.shape
        num_blocks_q  = math.ceil(N / Br)
        num_blocks_kv = math.ceil(N / Bc)

        # Initialize output matrix O (N, d)
        O = np.zeros_like(Q)

        # Statistics for validation and debugging
        m_history = []  # Track m_i per block
        l_history = []  # Track l_i per block

        if verbose:
            print(f"FlashAttention Configuration:")
            print(f"  Input shape: Q, K, V = ({N}, {d})")
            print(f"  Block sizes: Br={Br}, Bc={Bc}")
            print(f"  Num Q-blocks: {num_blocks_q}, Num KV-blocks: {num_blocks_kv}")
            print(f"  Scale factor: {scale:.6f}")
            print()

        # Algorithm 1: FlashAttention
        # Iterate over blocks of Q (queries)
        for i in range(num_blocks_q):
            # Extract Q block: [i*Br : min((i+1)*Br, N), :]
            q_start, q_end = i * Br, min((i + 1) * Br, N)
            Q_block = Q[q_start:q_end, :]  # (Br, d)

            # Initialize for this Q block:
            # m_i: running max of logits (shape: (Br,))
            # l_i: running denominator (shape: (Br,))
            # O_i: accumulator for output (shape: (Br, d))
            m_i = np.full((q_end - q_start,), -np.inf)
            l_i = np.zeros((q_end - q_start,))
            O_i = np.zeros((q_end - q_start, d))

            # Iterate over blocks of K, V
            for j in range(num_blocks_kv):
                kv_start, kv_end = j * Bc, min((j + 1) * Bc, N)
                K_block = K[kv_start:kv_end, :]  # (Bc, d)
                V_block = V[kv_start:kv_end, :]  # (Bc, d)

                # Compute block scores: Q_block @ K_block^T
                # [(Br, d) @ (d, Bc)] -> (Br, Bc)
                S_ij = Q_block @ K_block.T * scale

                # Online softmax update (Dao et al. Algorithm 1):

                # 1. Compute max of current block: m_ij = max(S_ij) per row
                m_ij_block = np.max(S_ij, axis=1)  # (Br,)

                # 2. New overall max: m_ij_new = max(m_i_old, m_ij_block)
                m_ij_new = np.maximum(m_i, m_ij_block)  # (Br,)

                # 3. Compute softmax for this block: P_ij = exp(S_ij - m_ij_new)
                P_ij = np.exp(S_ij - m_ij_new[:, np.newaxis])  # (Br, Bc)

                # 4. Compute sum of exponentials: l_ij_sum = sum(P_ij) per row
                l_ij_sum = np.sum(P_ij, axis=1)  # (Br,)

                # 5. Update running sum with rescaling for max change:
                #    l_ij_new = exp(m_i - m_ij_new) * l_i + l_ij_sum
                exp_factor = np.exp(m_i - m_ij_new)  # (Br,)
                l_ij_new = exp_factor * l_i + l_ij_sum  # (Br,)

                # 6. Update output O_i with rescaling:
                #    O_i_new = exp(m_i - m_ij_new) * O_i + P_ij @ V_block
                scale_factor = exp_factor[:, np.newaxis]  # (Br, 1)
                O_i = scale_factor * O_i + P_ij @ V_block  # (Br, d)

                # 7. Update running max and sum for next iteration
                m_i = m_ij_new
                l_i = l_ij_new

            # Final normalization: O_i = O_i / l_i
            O_i = O_i / l_i[:, np.newaxis]

            # Write back to output matrix
            O[q_start:q_end, :] = O_i

            m_history.append(m_i.copy())
            l_history.append(l_i.copy())

            if verbose and (i + 1) % max(1, num_blocks_q // 4) == 0:
                print(f"  Processed Q-block {i+1}/{num_blocks_q}")

        return O, {
            'algorithm': 'FlashAttention Algorithm 1',
            'block_sizes': {'Br': Br, 'Bc': Bc},
            'm_history': m_history,
            'l_history': l_history
        }

    def validate(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        Br: int = 32,
        Bc: int = 32,
        tolerance: float = 1e-4,
        verbose: bool = True
    ) -> Dict:
        """
        Compare FlashAttention against naive attention and verify correctness.

        Args:
            Q, K, V: Input matrices
            Br, Bc: Block sizes
            tolerance: Maximum allowed relative error
            verbose: Print detailed comparison

        Returns:
            results: Dict with error metrics and pass/fail status
        """
        scale = 1.0 / math.sqrt(Q.shape[-1])

        # Compute both versions
        O_naive, aux_naive = self.attention_naive(Q, K, V, scale)
        O_flash, aux_flash = self.attention_flash(Q, K, V, Br, Bc, scale, verbose=False)

        # Compute error metrics
        abs_error = np.abs(O_naive - O_flash)
        rel_error = np.abs(abs_error / (np.abs(O_naive) + self.eps))

        max_abs_error  = np.max(abs_error)
        max_rel_error  = np.max(rel_error)
        mean_abs_error = np.mean(abs_error)
        mean_rel_error = np.mean(rel_error)

        passed = max_rel_error < tolerance

        if verbose:
            print("\n" + "=" * 70)
            print("VALIDATION RESULTS")
            print("=" * 70)
            print(f"Input shape: {Q.shape}")
            print(f"Block config: Br={Br}, Bc={Bc}")
            print(f"Tolerance: {tolerance:.2e}")
            print()
            print("Error Metrics:")
            print(f"  Max Absolute Error:  {max_abs_error:.2e}")
            print(f"  Mean Absolute Error: {mean_abs_error:.2e}")
            print(f"  Max Relative Error:  {max_rel_error:.2e}")
            print(f"  Mean Relative Error: {mean_rel_error:.2e}")
            print()
            print(f"Status: {'PASS' if passed else 'FAIL'}")
            print("=" * 70 + "\n")

        return {
            'passed':          passed,
            'max_abs_error':   max_abs_error,
            'max_rel_error':   max_rel_error,
            'mean_abs_error':  mean_abs_error,
            'mean_rel_error':  mean_rel_error,
            'O_naive':         O_naive,
            'O_flash':         O_flash
        }


def run_golden_model_test():
    """
    Run a comprehensive golden model test with various configurations.
    """
    print("\n" + "=" * 70)
    print("FlashAttention RTL - Software Golden Model Test Suite")
    print("=" * 70 + "\n")

    # Start with a very simple test case
    print("Simple Sanity Check (N=4, d=2, Br=2, Bc=2):")
    print("-" * 70)
    np.random.seed(42)
    Q_simple = np.random.randn(4, 2).astype(np.float64) * 0.1
    K_simple = np.random.randn(4, 2).astype(np.float64) * 0.1
    V_simple = np.random.randn(4, 2).astype(np.float64) * 0.1

    model = FlashAttentionGoldenModel()
    results_simple = model.validate(Q_simple, K_simple, V_simple,
                                    Br=2, Bc=2, tolerance=1e-10, verbose=True)

    # Full test suite
    test_configs = [
        {'N':  64, 'd':  32, 'Br': 16, 'Bc': 16, 'name': 'Small  (N=64,  d=32)'},
        {'N': 128, 'd':  64, 'Br': 32, 'Bc': 32, 'name': 'Medium (N=128, d=64)'},
        {'N': 256, 'd': 128, 'Br': 64, 'Bc': 64, 'name': 'Large  (N=256, d=128)'},
    ]

    all_passed = True

    for config in test_configs:
        N    = config['N']
        d    = config['d']
        Br   = config['Br']
        Bc   = config['Bc']
        name = config['name']

        print(f"\nTest: {name}")
        print("-" * 70)

        # Generate random data (seed for reproducibility) -- use float64 for golden model
        np.random.seed(42)
        Q = np.random.randn(N, d).astype(np.float64) * 0.1
        K = np.random.randn(N, d).astype(np.float64) * 0.1
        V = np.random.randn(N, d).astype(np.float64) * 0.1

        # Validate with higher precision
        results = model.validate(Q, K, V, Br=Br, Bc=Bc, tolerance=1e-5, verbose=True)
        all_passed = all_passed and results['passed']

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    run_golden_model_test()
