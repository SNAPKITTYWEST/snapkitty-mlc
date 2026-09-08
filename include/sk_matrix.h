/* sk_matrix.h — SnapKitty MLC: row-major f32 matrix operations
 *
 * All operations take an explicit arena for allocation.
 * Forward pass and gradient ops are separated.
 * No hidden globals; all state is in the passed structs.
 *
 * Authors: Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)
 * Source:  SNAPKITTYAGENT9NOVA/MLC
 * License: BSL-1.1 / AGPL-3.0 / MPL-2.0
 */
#ifndef SK_MATRIX_H
#define SK_MATRIX_H

#include "sk_defs.h"
#include "sk_arena.h"

/* ── Matrix ───────────────────────────────────────────────────────── */
typedef struct {
    u32  rows;
    u32  cols;
    f32* data;   /* row-major, size = rows × cols × sizeof(f32) */
} sk_matrix;

/* ── Lifecycle ────────────────────────────────────────────────────── */
sk_matrix* sk_mat_create(sk_arena* arena, u32 rows, u32 cols);
sk_matrix* sk_mat_load(sk_arena* arena, u32 rows, u32 cols, const char* filename);
b32        sk_mat_copy(sk_matrix* dst, const sk_matrix* src);

/* ── Fill ─────────────────────────────────────────────────────────── */
void  sk_mat_clear(sk_matrix* mat);
void  sk_mat_fill(sk_matrix* mat, f32 x);
void  sk_mat_fill_rand(sk_matrix* mat, f32 lower, f32 upper);

/* ── Reductions ───────────────────────────────────────────────────── */
void  sk_mat_scale(sk_matrix* mat, f32 scale);
f32   sk_mat_sum(const sk_matrix* mat);
u64   sk_mat_argmax(const sk_matrix* mat);

/* ── Element-wise arithmetic ──────────────────────────────────────── */
b32   sk_mat_add(sk_matrix* out, const sk_matrix* a, const sk_matrix* b);
b32   sk_mat_sub(sk_matrix* out, const sk_matrix* a, const sk_matrix* b);

/* ── Matrix multiply
 *   out = A(ᵀ?) × B(ᵀ?)   zero_out: clear out before accumulating
 *   transpose_a/b: treat operand as its transpose (no copy)
 * ─────────────────────────────────────────────────────────────────── */
b32   sk_mat_mul(
    sk_matrix* out, const sk_matrix* a, const sk_matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);

/* ── Activation functions (forward) ──────────────────────────────── */
b32   sk_mat_relu(sk_matrix* out, const sk_matrix* in);
b32   sk_mat_softmax(sk_matrix* out, const sk_matrix* in);

/* ── Loss function ────────────────────────────────────────────────── */
b32   sk_mat_cross_entropy(sk_matrix* out, const sk_matrix* p, const sk_matrix* q);

/* ── Gradient accumulation (adds into out, never overwrites) ─────── */
b32   sk_mat_relu_add_grad(
    sk_matrix* out, const sk_matrix* in, const sk_matrix* grad
);
b32   sk_mat_softmax_add_grad(
    sk_matrix* out, const sk_matrix* softmax_out, const sk_matrix* grad
);
b32   sk_mat_cross_entropy_add_grad(
    sk_matrix* p_grad, sk_matrix* q_grad,
    const sk_matrix* p, const sk_matrix* q, const sk_matrix* grad
);

/* Compatibility aliases */
#define matrix                    sk_matrix
#define mat_create                sk_mat_create
#define mat_load                  sk_mat_load
#define mat_copy                  sk_mat_copy
#define mat_clear                 sk_mat_clear
#define mat_fill                  sk_mat_fill
#define mat_fill_rand             sk_mat_fill_rand
#define mat_scale                 sk_mat_scale
#define mat_sum                   sk_mat_sum
#define mat_argmax                sk_mat_argmax
#define mat_add                   sk_mat_add
#define mat_sub                   sk_mat_sub
#define mat_mul                   sk_mat_mul
#define mat_relu                  sk_mat_relu
#define mat_softmax               sk_mat_softmax
#define mat_cross_entropy         sk_mat_cross_entropy
#define mat_relu_add_grad         sk_mat_relu_add_grad
#define mat_softmax_add_grad      sk_mat_softmax_add_grad
#define mat_cross_entropy_add_grad sk_mat_cross_entropy_add_grad

#endif /* SK_MATRIX_H */
