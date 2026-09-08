/* sk_model.h — SnapKitty MLC: computation graph, autograd, training
 *
 * Design:
 *   - All allocation is arena-based (no malloc/free in the hot path)
 *   - Computation graph is built by composing sk_mv_* constructors
 *   - Topological sort in sk_model_prog_create → forward/backward in O(n)
 *   - Training: mini-batch SGD with Fisher-Yates shuffle per epoch
 *
 * Authors: Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)
 * Source:  SNAPKITTYAGENT9NOVA/MLC
 * License: BSL-1.1 / AGPL-3.0 / MPL-2.0
 */
#ifndef SK_MODEL_H
#define SK_MODEL_H

#include "sk_defs.h"
#include "sk_arena.h"
#include "sk_matrix.h"

/* ── Variable flags ───────────────────────────────────────────────── */
typedef enum {
    SK_MV_NONE           = 0,
    SK_MV_REQUIRES_GRAD  = (1 << 0),  /* accumulate gradients */
    SK_MV_PARAMETER      = (1 << 1),  /* updated during training */
    SK_MV_INPUT          = (1 << 2),  /* model input slot */
    SK_MV_OUTPUT         = (1 << 3),  /* model output slot */
    SK_MV_DESIRED_OUTPUT = (1 << 4),  /* ground-truth label slot */
    SK_MV_COST           = (1 << 5),  /* scalar loss slot */
} sk_mv_flags;

/* ── Operation type ───────────────────────────────────────────────── */
typedef enum {
    SK_OP_NULL = 0,
    SK_OP_CREATE,
    _SK_OP_UNARY_START,
    SK_OP_RELU,
    SK_OP_SOFTMAX,
    _SK_OP_BINARY_START,
    SK_OP_ADD,
    SK_OP_SUB,
    SK_OP_MATMUL,
    SK_OP_CROSS_ENTROPY,
} sk_mv_op;

#define SK_MV_MAX_INPUTS 2
#define SK_MV_NUM_INPUTS(op) \
    ((op) < _SK_OP_UNARY_START ? 0 : ((op) < _SK_OP_BINARY_START ? 1 : 2))

/* ── Model variable (node in the computation graph) ──────────────── */
typedef struct sk_model_var {
    u32    index;
    u32    flags;
    sk_matrix* val;
    sk_matrix* grad;
    sk_mv_op   op;
    struct sk_model_var* inputs[SK_MV_MAX_INPUTS];
} sk_model_var;

/* ── Ordered execution program (topological sort output) ─────────── */
typedef struct {
    sk_model_var** vars;
    u32            size;
} sk_model_prog;

/* ── Model context ────────────────────────────────────────────────── */
typedef struct {
    u32 num_vars;
    sk_model_var* input;
    sk_model_var* output;
    sk_model_var* desired_output;
    sk_model_var* cost;
    sk_model_prog forward_prog;
    sk_model_prog cost_prog;
} sk_model;

/* ── Training descriptor ──────────────────────────────────────────── */
typedef struct {
    sk_matrix* train_images;
    sk_matrix* train_labels;
    sk_matrix* test_images;
    sk_matrix* test_labels;
    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} sk_training_desc;

/* ── Variable constructors ────────────────────────────────────────── */
sk_model_var* sk_mv_create(
    sk_arena* arena, sk_model* model, u32 rows, u32 cols, u32 flags
);
sk_model_var* sk_mv_relu(
    sk_arena* arena, sk_model* model, sk_model_var* input, u32 flags
);
sk_model_var* sk_mv_softmax(
    sk_arena* arena, sk_model* model, sk_model_var* input, u32 flags
);
sk_model_var* sk_mv_add(
    sk_arena* arena, sk_model* model,
    sk_model_var* a, sk_model_var* b, u32 flags
);
sk_model_var* sk_mv_sub(
    sk_arena* arena, sk_model* model,
    sk_model_var* a, sk_model_var* b, u32 flags
);
sk_model_var* sk_mv_matmul(
    sk_arena* arena, sk_model* model,
    sk_model_var* a, sk_model_var* b, u32 flags
);
sk_model_var* sk_mv_cross_entropy(
    sk_arena* arena, sk_model* model,
    sk_model_var* p, sk_model_var* q, u32 flags
);

/* ── Program execution ────────────────────────────────────────────── */
sk_model_prog sk_model_prog_create(
    sk_arena* arena, sk_model* model, sk_model_var* out_var
);
void sk_model_prog_compute(sk_model_prog* prog);
void sk_model_prog_compute_grads(sk_model_prog* prog);

/* ── Model lifecycle ──────────────────────────────────────────────── */
sk_model* sk_model_create(sk_arena* arena);
void      sk_model_compile(sk_arena* arena, sk_model* model);
void      sk_model_feedforward(sk_model* model);
void      sk_model_train(sk_model* model, const sk_training_desc* desc);

/* Compatibility aliases (original names) */
#define MV_FLAG_NONE           SK_MV_NONE
#define MV_FLAG_REQUIRES_GRAD  SK_MV_REQUIRES_GRAD
#define MV_FLAG_PARAMETER      SK_MV_PARAMETER
#define MV_FLAG_INPUT          SK_MV_INPUT
#define MV_FLAG_OUTPUT         SK_MV_OUTPUT
#define MV_FLAG_DESIRED_OUTPUT SK_MV_DESIRED_OUTPUT
#define MV_FLAG_COST           SK_MV_COST
#define MV_OP_NULL           SK_OP_NULL
#define MV_OP_CREATE         SK_OP_CREATE
#define _MV_OP_UNARY_START   _SK_OP_UNARY_START
#define MV_OP_RELU           SK_OP_RELU
#define MV_OP_SOFTMAX        SK_OP_SOFTMAX
#define _MV_OP_BINARY_START  _SK_OP_BINARY_START
#define MV_OP_ADD            SK_OP_ADD
#define MV_OP_SUB            SK_OP_SUB
#define MV_OP_MATMUL         SK_OP_MATMUL
#define MV_OP_CROSS_ENTROPY  SK_OP_CROSS_ENTROPY
#define MODEL_VAR_MAX_INPUTS SK_MV_MAX_INPUTS
#define MV_NUM_INPUTS        SK_MV_NUM_INPUTS
#define model_var            sk_model_var
#define model_var_flags      sk_mv_flags
#define model_var_op         sk_mv_op
#define model_program        sk_model_prog
#define model_context        sk_model
#define model_training_desc  sk_training_desc
#define mv_create            sk_mv_create
#define mv_relu              sk_mv_relu
#define mv_softmax           sk_mv_softmax
#define mv_add               sk_mv_add
#define mv_sub               sk_mv_sub
#define mv_matmul            sk_mv_matmul
#define mv_cross_entropy     sk_mv_cross_entropy
#define model_prog_create        sk_model_prog_create
#define model_prog_compute       sk_model_prog_compute
#define model_prog_compute_grads sk_model_prog_compute_grads
#define model_create             sk_model_create
#define model_compile            sk_model_compile
#define model_feedforward        sk_model_feedforward
#define model_train              sk_model_train

#endif /* SK_MODEL_H */
