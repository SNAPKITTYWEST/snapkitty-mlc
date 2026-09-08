/* sk_matrix.c — SnapKitty MLC: matrix operations
 *
 * Authors: Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)
 * Source:  SNAPKITTYAGENT9NOVA/MLC
 * License: BSL-1.1 / AGPL-3.0 / MPL-2.0
 */

#define _CRT_SECURE_NO_WARNINGS

#include "../include/sk_defs.h"
#include "../include/sk_arena.h"
#include "../include/sk_random.h"
#include "../include/sk_matrix.h"

/* ── Lifecycle ────────────────────────────────────────────────────── */

sk_matrix* sk_mat_create(sk_arena* arena, u32 rows, u32 cols) {
    sk_matrix* mat = SK_PUSH_STRUCT(arena, sk_matrix);
    mat->rows = rows;
    mat->cols = cols;
    mat->data = SK_PUSH_ARRAY(arena, f32, (u64)rows * cols);
    return mat;
}

sk_matrix* sk_mat_load(sk_arena* arena, u32 rows, u32 cols, const char* filename) {
    sk_matrix* mat = sk_mat_create(arena, rows, cols);
    FILE* f = fopen(filename, "rb");
    if (!f) return mat;
    fseek(f, 0, SEEK_END);
    u64 size = (u64)ftell(f);
    fseek(f, 0, SEEK_SET);
    size = SK_MIN(size, sizeof(f32) * (u64)rows * cols);
    fread(mat->data, 1, size, f);
    fclose(f);
    return mat;
}

b32 sk_mat_copy(sk_matrix* dst, const sk_matrix* src) {
    if (dst->rows != src->rows || dst->cols != src->cols) return false;
    memcpy(dst->data, src->data, sizeof(f32) * (u64)dst->rows * dst->cols);
    return true;
}

/* ── Fill ─────────────────────────────────────────────────────────── */

void sk_mat_clear(sk_matrix* mat) {
    memset(mat->data, 0, sizeof(f32) * (u64)mat->rows * mat->cols);
}

void sk_mat_fill(sk_matrix* mat, f32 x) {
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) mat->data[i] = x;
}

void sk_mat_fill_rand(sk_matrix* mat, f32 lower, f32 upper) {
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++)
        mat->data[i] = sk_prng_randf() * (upper - lower) + lower;
}

/* ── Reductions ───────────────────────────────────────────────────── */

void sk_mat_scale(sk_matrix* mat, f32 scale) {
    u64 size = (u64)mat->rows * mat->cols;
    for (u64 i = 0; i < size; i++) mat->data[i] *= scale;
}

f32 sk_mat_sum(const sk_matrix* mat) {
    u64 size = (u64)mat->rows * mat->cols;
    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) sum += mat->data[i];
    return sum;
}

u64 sk_mat_argmax(const sk_matrix* mat) {
    u64 size = (u64)mat->rows * mat->cols;
    u64 max_i = 0;
    for (u64 i = 1; i < size; i++)
        if (mat->data[i] > mat->data[max_i]) max_i = i;
    return max_i;
}

/* ── Element-wise arithmetic ──────────────────────────────────────── */

b32 sk_mat_add(sk_matrix* out, const sk_matrix* a, const sk_matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) return false;
    if (out->rows != a->rows || out->cols != a->cols) return false;
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) out->data[i] = a->data[i] + b->data[i];
    return true;
}

b32 sk_mat_sub(sk_matrix* out, const sk_matrix* a, const sk_matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) return false;
    if (out->rows != a->rows || out->cols != a->cols) return false;
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) out->data[i] = a->data[i] - b->data[i];
    return true;
}

/* ── Matrix multiply (four transpose variants) ───────────────────── */

static void _mat_mul_nn(sk_matrix* out, const sk_matrix* a, const sk_matrix* b) {
    for (u64 i = 0; i < out->rows; i++)
        for (u64 k = 0; k < a->cols; k++)
            for (u64 j = 0; j < out->cols; j++)
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] * b->data[j + k * b->cols];
}

static void _mat_mul_nt(sk_matrix* out, const sk_matrix* a, const sk_matrix* b) {
    for (u64 i = 0; i < out->rows; i++)
        for (u64 j = 0; j < out->cols; j++)
            for (u64 k = 0; k < a->cols; k++)
                out->data[j + i * out->cols] +=
                    a->data[k + i * a->cols] * b->data[k + j * b->cols];
}

static void _mat_mul_tn(sk_matrix* out, const sk_matrix* a, const sk_matrix* b) {
    for (u64 k = 0; k < a->rows; k++)
        for (u64 i = 0; i < out->rows; i++)
            for (u64 j = 0; j < out->cols; j++)
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] * b->data[j + k * b->cols];
}

static void _mat_mul_tt(sk_matrix* out, const sk_matrix* a, const sk_matrix* b) {
    for (u64 i = 0; i < out->rows; i++)
        for (u64 j = 0; j < out->cols; j++)
            for (u64 k = 0; k < a->rows; k++)
                out->data[j + i * out->cols] +=
                    a->data[i + k * a->cols] * b->data[k + j * b->cols];
}

b32 sk_mat_mul(
    sk_matrix* out, const sk_matrix* a, const sk_matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
) {
    u32 a_rows = transpose_a ? a->cols : a->rows;
    u32 a_cols = transpose_a ? a->rows : a->cols;
    u32 b_rows = transpose_b ? b->cols : b->rows;
    u32 b_cols = transpose_b ? b->rows : b->cols;
    if (a_cols != b_rows) return false;
    if (out->rows != a_rows || out->cols != b_cols) return false;
    if (zero_out) sk_mat_clear(out);
    switch ((transpose_a << 1) | transpose_b) {
        case 0: _mat_mul_nn(out, a, b); break;
        case 1: _mat_mul_nt(out, a, b); break;
        case 2: _mat_mul_tn(out, a, b); break;
        case 3: _mat_mul_tt(out, a, b); break;
    }
    return true;
}

/* ── Activations ──────────────────────────────────────────────────── */

b32 sk_mat_relu(sk_matrix* out, const sk_matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) return false;
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++) out->data[i] = SK_MAX(0, in->data[i]);
    return true;
}

b32 sk_mat_softmax(sk_matrix* out, const sk_matrix* in) {
    if (out->rows != in->rows || out->cols != in->cols) return false;
    u64 size = (u64)out->rows * out->cols;
    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) { out->data[i] = expf(in->data[i]); sum += out->data[i]; }
    sk_mat_scale(out, 1.0f / sum);
    return true;
}

b32 sk_mat_cross_entropy(sk_matrix* out, const sk_matrix* p, const sk_matrix* q) {
    if (p->rows != q->rows || p->cols != q->cols) return false;
    if (out->rows != p->rows || out->cols != p->cols) return false;
    u64 size = (u64)p->rows * p->cols;
    for (u64 i = 0; i < size; i++)
        out->data[i] = p->data[i] == 0.0f ? 0.0f : p->data[i] * -logf(q->data[i]);
    return true;
}

/* ── Gradient accumulation ────────────────────────────────────────── */

b32 sk_mat_relu_add_grad(sk_matrix* out, const sk_matrix* in, const sk_matrix* grad) {
    if (out->rows != in->rows || out->cols != in->cols) return false;
    if (out->rows != grad->rows || out->cols != grad->cols) return false;
    u64 size = (u64)out->rows * out->cols;
    for (u64 i = 0; i < size; i++)
        out->data[i] += in->data[i] > 0.0f ? grad->data[i] : 0.0f;
    return true;
}

b32 sk_mat_softmax_add_grad(
    sk_matrix* out, const sk_matrix* softmax_out, const sk_matrix* grad
) {
    if (softmax_out->rows != 1 && softmax_out->cols != 1) return false;
    sk_arena_temp scratch = sk_arena_scratch_get(NULL, 0);
    u32 size = SK_MAX(softmax_out->rows, softmax_out->cols);
    sk_matrix* jacobian = sk_mat_create(scratch.arena, size, size);
    for (u32 i = 0; i < size; i++)
        for (u32 j = 0; j < size; j++)
            jacobian->data[j + i * size] =
                softmax_out->data[i] * ((i == j) - softmax_out->data[j]);
    sk_mat_mul(out, jacobian, grad, 0, 0, 0);
    sk_arena_scratch_release(scratch);
    return true;
}

b32 sk_mat_cross_entropy_add_grad(
    sk_matrix* p_grad, sk_matrix* q_grad,
    const sk_matrix* p, const sk_matrix* q, const sk_matrix* grad
) {
    if (p->rows != q->rows || p->cols != q->cols) return false;
    u64 size = (u64)p->rows * p->cols;
    if (p_grad) {
        if (p_grad->rows != p->rows || p_grad->cols != p->cols) return false;
        for (u64 i = 0; i < size; i++)
            p_grad->data[i] += -logf(q->data[i]) * grad->data[i];
    }
    if (q_grad) {
        if (q_grad->rows != q->rows || q_grad->cols != q->cols) return false;
        for (u64 i = 0; i < size; i++)
            q_grad->data[i] += -p->data[i] / q->data[i] * grad->data[i];
    }
    return true;
}
