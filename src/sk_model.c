/* sk_model.c — SnapKitty MLC: computation graph, autograd, training loop
 *
 * Authors: Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)
 * Source:  SNAPKITTYAGENT9NOVA/MLC
 * License: BSL-1.1 / AGPL-3.0 / MPL-2.0
 */

#include "../include/sk_defs.h"
#include "../include/sk_arena.h"
#include "../include/sk_random.h"
#include "../include/sk_matrix.h"
#include "../include/sk_model.h"

/* ── Internal helpers ─────────────────────────────────────────────── */

static sk_model_var* _mv_unary_impl(
    sk_arena* arena, sk_model* model,
    sk_model_var* input, u32 rows, u32 cols,
    u32 flags, sk_mv_op op
) {
    if (input->flags & SK_MV_REQUIRES_GRAD) flags |= SK_MV_REQUIRES_GRAD;
    sk_model_var* out = sk_mv_create(arena, model, rows, cols, flags);
    out->op = op;
    out->inputs[0] = input;
    return out;
}

static sk_model_var* _mv_binary_impl(
    sk_arena* arena, sk_model* model,
    sk_model_var* a, sk_model_var* b,
    u32 rows, u32 cols, u32 flags, sk_mv_op op
) {
    if ((a->flags & SK_MV_REQUIRES_GRAD) || (b->flags & SK_MV_REQUIRES_GRAD))
        flags |= SK_MV_REQUIRES_GRAD;
    sk_model_var* out = sk_mv_create(arena, model, rows, cols, flags);
    out->op = op;
    out->inputs[0] = a;
    out->inputs[1] = b;
    return out;
}

/* ── Variable constructors ────────────────────────────────────────── */

sk_model_var* sk_mv_create(
    sk_arena* arena, sk_model* model, u32 rows, u32 cols, u32 flags
) {
    sk_model_var* out = SK_PUSH_STRUCT(arena, sk_model_var);
    out->index = model->num_vars++;
    out->flags = flags;
    out->op    = SK_OP_CREATE;
    out->val   = sk_mat_create(arena, rows, cols);
    if (flags & SK_MV_REQUIRES_GRAD)
        out->grad = sk_mat_create(arena, rows, cols);
    if (flags & SK_MV_INPUT)          model->input          = out;
    if (flags & SK_MV_OUTPUT)         model->output         = out;
    if (flags & SK_MV_DESIRED_OUTPUT) model->desired_output = out;
    if (flags & SK_MV_COST)           model->cost           = out;
    return out;
}

sk_model_var* sk_mv_relu(
    sk_arena* arena, sk_model* model, sk_model_var* input, u32 flags
) {
    return _mv_unary_impl(arena, model, input,
        input->val->rows, input->val->cols, flags, SK_OP_RELU);
}

sk_model_var* sk_mv_softmax(
    sk_arena* arena, sk_model* model, sk_model_var* input, u32 flags
) {
    return _mv_unary_impl(arena, model, input,
        input->val->rows, input->val->cols, flags, SK_OP_SOFTMAX);
}

sk_model_var* sk_mv_add(
    sk_arena* arena, sk_model* model,
    sk_model_var* a, sk_model_var* b, u32 flags
) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) return NULL;
    return _mv_binary_impl(arena, model, a, b,
        a->val->rows, a->val->cols, flags, SK_OP_ADD);
}

sk_model_var* sk_mv_sub(
    sk_arena* arena, sk_model* model,
    sk_model_var* a, sk_model_var* b, u32 flags
) {
    if (a->val->rows != b->val->rows || a->val->cols != b->val->cols) return NULL;
    return _mv_binary_impl(arena, model, a, b,
        a->val->rows, a->val->cols, flags, SK_OP_SUB);
}

sk_model_var* sk_mv_matmul(
    sk_arena* arena, sk_model* model,
    sk_model_var* a, sk_model_var* b, u32 flags
) {
    if (a->val->cols != b->val->rows) return NULL;
    return _mv_binary_impl(arena, model, a, b,
        a->val->rows, b->val->cols, flags, SK_OP_MATMUL);
}

sk_model_var* sk_mv_cross_entropy(
    sk_arena* arena, sk_model* model,
    sk_model_var* p, sk_model_var* q, u32 flags
) {
    if (p->val->rows != q->val->rows || p->val->cols != q->val->cols) return NULL;
    return _mv_binary_impl(arena, model, p, q,
        p->val->rows, p->val->cols, flags, SK_OP_CROSS_ENTROPY);
}

/* ── Program: topological sort (iterative DFS) ───────────────────── */

sk_model_prog sk_model_prog_create(
    sk_arena* arena, sk_model* model, sk_model_var* out_var
) {
    sk_arena_temp scratch = sk_arena_scratch_get(&arena, 1);

    b8*            visited    = SK_PUSH_ARRAY(scratch.arena, b8, model->num_vars);
    sk_model_var** stack      = SK_PUSH_ARRAY(scratch.arena, sk_model_var*, model->num_vars);
    sk_model_var** out        = SK_PUSH_ARRAY(scratch.arena, sk_model_var*, model->num_vars);
    u32 stack_size = 0, out_size = 0;

    stack[stack_size++] = out_var;

    while (stack_size > 0) {
        sk_model_var* cur = stack[--stack_size];
        if (cur->index >= model->num_vars) continue;
        if (visited[cur->index]) {
            if (out_size < model->num_vars) out[out_size++] = cur;
            continue;
        }
        visited[cur->index] = true;
        if (stack_size < model->num_vars) stack[stack_size++] = cur;
        u32 n = SK_MV_NUM_INPUTS(cur->op);
        for (u32 i = 0; i < n; i++) {
            sk_model_var* inp = cur->inputs[i];
            if (!inp || inp->index >= model->num_vars || visited[inp->index]) continue;
            /* remove duplicates in stack */
            for (u32 j = 0; j < stack_size; j++) {
                if (stack[j] == inp) {
                    for (u32 k = j; k < stack_size - 1; k++) stack[k] = stack[k+1];
                    stack_size--;
                    break;
                }
            }
            if (stack_size < model->num_vars) stack[stack_size++] = inp;
        }
    }

    sk_model_prog prog = {
        .size = out_size,
        .vars = SK_PUSH_ARRAY_NZ(arena, sk_model_var*, out_size)
    };
    memcpy(prog.vars, out, sizeof(sk_model_var*) * out_size);
    sk_arena_scratch_release(scratch);
    return prog;
}

/* ── Forward pass ─────────────────────────────────────────────────── */

void sk_model_prog_compute(sk_model_prog* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        sk_model_var* cur = prog->vars[i];
        sk_model_var* a   = cur->inputs[0];
        sk_model_var* b   = cur->inputs[1];
        switch (cur->op) {
            case SK_OP_NULL:
            case SK_OP_CREATE: break;
            case _SK_OP_UNARY_START: break;
            case SK_OP_RELU:         sk_mat_relu(cur->val, a->val); break;
            case SK_OP_SOFTMAX:      sk_mat_softmax(cur->val, a->val); break;
            case _SK_OP_BINARY_START: break;
            case SK_OP_ADD:         sk_mat_add(cur->val, a->val, b->val); break;
            case SK_OP_SUB:         sk_mat_sub(cur->val, a->val, b->val); break;
            case SK_OP_MATMUL:      sk_mat_mul(cur->val, a->val, b->val, 1, 0, 0); break;
            case SK_OP_CROSS_ENTROPY: sk_mat_cross_entropy(cur->val, a->val, b->val); break;
        }
    }
}

/* ── Backward pass (reverse-mode autodiff) ───────────────────────── */

void sk_model_prog_compute_grads(sk_model_prog* prog) {
    /* Zero non-parameter grads */
    for (u32 i = 0; i < prog->size; i++) {
        sk_model_var* cur = prog->vars[i];
        if ((cur->flags & SK_MV_REQUIRES_GRAD) && !(cur->flags & SK_MV_PARAMETER))
            sk_mat_clear(cur->grad);
    }
    /* Seed loss gradient = 1 */
    sk_mat_fill(prog->vars[prog->size - 1]->grad, 1.0f);

    for (i64 i = (i64)prog->size - 1; i >= 0; i--) {
        sk_model_var* cur = prog->vars[i];
        if (!(cur->flags & SK_MV_REQUIRES_GRAD)) continue;
        sk_model_var* a = cur->inputs[0];
        sk_model_var* b = cur->inputs[1];
        u32 n = SK_MV_NUM_INPUTS(cur->op);
        if (n == 1 && !(a->flags & SK_MV_REQUIRES_GRAD)) continue;
        if (n == 2 &&
            !(a->flags & SK_MV_REQUIRES_GRAD) &&
            !(b->flags & SK_MV_REQUIRES_GRAD)) continue;

        switch (cur->op) {
            case SK_OP_NULL:
            case SK_OP_CREATE: break;
            case _SK_OP_UNARY_START: break;
            case SK_OP_RELU:
                sk_mat_relu_add_grad(a->grad, a->val, cur->grad); break;
            case SK_OP_SOFTMAX:
                sk_mat_softmax_add_grad(a->grad, cur->val, cur->grad); break;
            case _SK_OP_BINARY_START: break;
            case SK_OP_ADD:
                if (a->flags & SK_MV_REQUIRES_GRAD) sk_mat_add(a->grad, a->grad, cur->grad);
                if (b->flags & SK_MV_REQUIRES_GRAD) sk_mat_add(b->grad, b->grad, cur->grad);
                break;
            case SK_OP_SUB:
                if (a->flags & SK_MV_REQUIRES_GRAD) sk_mat_add(a->grad, a->grad, cur->grad);
                if (b->flags & SK_MV_REQUIRES_GRAD) sk_mat_sub(b->grad, b->grad, cur->grad);
                break;
            case SK_OP_MATMUL:
                if (a->flags & SK_MV_REQUIRES_GRAD) sk_mat_mul(a->grad, cur->grad, b->val, 0, 0, 1);
                if (b->flags & SK_MV_REQUIRES_GRAD) sk_mat_mul(b->grad, a->val, cur->grad, 0, 1, 0);
                break;
            case SK_OP_CROSS_ENTROPY:
                sk_mat_cross_entropy_add_grad(
                    a->grad, b->grad, a->val, b->val, cur->grad); break;
        }
    }
}

/* ── Model lifecycle ──────────────────────────────────────────────── */

sk_model* sk_model_create(sk_arena* arena) {
    return SK_PUSH_STRUCT(arena, sk_model);
}

void sk_model_compile(sk_arena* arena, sk_model* model) {
    if (model->output)
        model->forward_prog = sk_model_prog_create(arena, model, model->output);
    if (model->cost)
        model->cost_prog = sk_model_prog_create(arena, model, model->cost);
}

void sk_model_feedforward(sk_model* model) {
    sk_model_prog_compute(&model->forward_prog);
}

/* ── Training: mini-batch SGD ─────────────────────────────────────── */

void sk_model_train(sk_model* model, const sk_training_desc* desc) {
    u32 num_examples = desc->train_images->rows;
    u32 input_size   = desc->train_images->cols;
    u32 output_size  = desc->train_labels->cols;
    u32 num_tests    = desc->test_images->rows;
    u32 num_batches  = num_examples / desc->batch_size;

    sk_arena_temp scratch = sk_arena_scratch_get(NULL, 0);
    u32* order = SK_PUSH_ARRAY_NZ(scratch.arena, u32, num_examples);
    for (u32 i = 0; i < num_examples; i++) order[i] = i;

    for (u32 epoch = 0; epoch < desc->epochs; epoch++) {
        /* Fisher-Yates shuffle */
        for (u32 i = 0; i < num_examples; i++) {
            u32 a = sk_prng_rand() % num_examples;
            u32 b2 = sk_prng_rand() % num_examples;
            u32 tmp = order[b2]; order[b2] = order[a]; order[a] = tmp;
        }

        for (u32 batch = 0; batch < num_batches; batch++) {
            /* Zero parameter grads */
            for (u32 i = 0; i < model->cost_prog.size; i++) {
                sk_model_var* cur = model->cost_prog.vars[i];
                if (cur->flags & SK_MV_PARAMETER) sk_mat_clear(cur->grad);
            }

            f32 avg_cost = 0.0f;
            for (u32 i = 0; i < desc->batch_size; i++) {
                u32 idx = order[batch * desc->batch_size + i];
                memcpy(model->input->val->data,
                       desc->train_images->data + idx * input_size,
                       sizeof(f32) * input_size);
                memcpy(model->desired_output->val->data,
                       desc->train_labels->data + idx * output_size,
                       sizeof(f32) * output_size);
                sk_model_prog_compute(&model->cost_prog);
                sk_model_prog_compute_grads(&model->cost_prog);
                avg_cost += sk_mat_sum(model->cost->val);
            }
            avg_cost /= (f32)desc->batch_size;

            /* SGD update */
            for (u32 i = 0; i < model->cost_prog.size; i++) {
                sk_model_var* cur = model->cost_prog.vars[i];
                if (!(cur->flags & SK_MV_PARAMETER)) continue;
                sk_mat_scale(cur->grad, desc->learning_rate / desc->batch_size);
                sk_mat_sub(cur->val, cur->val, cur->grad);
            }

            printf("Epoch %2d/%2d  Batch %4d/%4d  Cost: %.4f\r",
                   epoch+1, desc->epochs, batch+1, num_batches, avg_cost);
            fflush(stdout);
        }
        printf("\n");

        /* Epoch evaluation */
        u32 num_correct = 0;
        f32 avg_cost = 0.0f;
        for (u32 i = 0; i < num_tests; i++) {
            memcpy(model->input->val->data,
                   desc->test_images->data + i * input_size,
                   sizeof(f32) * input_size);
            memcpy(model->desired_output->val->data,
                   desc->test_labels->data + i * output_size,
                   sizeof(f32) * output_size);
            sk_model_prog_compute(&model->cost_prog);
            avg_cost += sk_mat_sum(model->cost->val);
            num_correct += (sk_mat_argmax(model->output->val) ==
                            sk_mat_argmax(model->desired_output->val));
        }
        avg_cost /= (f32)num_tests;
        printf("Test  Accuracy: %5u/%5u (%.1f%%)  Cost: %.4f\n",
               num_correct, num_tests,
               (f32)num_correct / num_tests * 100.0f, avg_cost);
    }

    sk_arena_scratch_release(scratch);
}
