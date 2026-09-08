/* mnist.c — SnapKitty MLC: MNIST digit classifier example
 *
 * Architecture:
 *   Input (784) → W0(16×784) ReLU → W1(16×16) ReLU [+ skip] → W2(10×16) Softmax
 *   Loss: cross-entropy
 *   Init: Xavier uniform (Glorot)
 *   Train: SGD, 10 epochs, batch=50, lr=0.01
 *
 * Data: run examples/data_convert.py first to generate .mat files.
 *
 * Authors: Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)
 * Source:  SNAPKITTYAGENT9NOVA/MLC
 * License: BSL-1.1 / AGPL-3.0 / MPL-2.0
 */

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "../include/sk_defs.h"
#include "../include/sk_arena.h"
#include "../include/sk_random.h"
#include "../include/sk_matrix.h"
#include "../include/sk_model.h"

/* ── ANSI terminal: render digit as coloured blocks ──────────────── */
static void draw_mnist_digit(const f32* data) {
    for (u32 y = 0; y < 28; y++) {
        for (u32 x = 0; x < 28; x++) {
            f32 num = data[x + y * 28];
            u32 col = 232 + (u32)(num * 23);
            printf("\x1b[48;5;%dm  ", col);
        }
        printf("\x1b[0m\n");
    }
    printf("\x1b[0m");
}

/* ── Build the MNIST model ────────────────────────────────────────── */
static void create_mnist_model(sk_arena* arena, sk_model* model) {
    /* Xavier uniform init bounds */
    f32 b0 = sqrtf(6.0f / (784 + 16));
    f32 b1 = sqrtf(6.0f / ( 16 + 16));
    f32 b2 = sqrtf(6.0f / ( 16 + 10));

    sk_model_var* input = sk_mv_create(arena, model, 784, 1, SK_MV_INPUT);

    sk_model_var* W0 = sk_mv_create(arena, model, 16, 784, SK_MV_REQUIRES_GRAD | SK_MV_PARAMETER);
    sk_model_var* W1 = sk_mv_create(arena, model, 16,  16, SK_MV_REQUIRES_GRAD | SK_MV_PARAMETER);
    sk_model_var* W2 = sk_mv_create(arena, model, 10,  16, SK_MV_REQUIRES_GRAD | SK_MV_PARAMETER);
    sk_mat_fill_rand(W0->val, -b0, b0);
    sk_mat_fill_rand(W1->val, -b1, b1);
    sk_mat_fill_rand(W2->val, -b2, b2);

    sk_model_var* b0v = sk_mv_create(arena, model, 16, 1, SK_MV_REQUIRES_GRAD | SK_MV_PARAMETER);
    sk_model_var* b1v = sk_mv_create(arena, model, 16, 1, SK_MV_REQUIRES_GRAD | SK_MV_PARAMETER);
    sk_model_var* b2v = sk_mv_create(arena, model, 10, 1, SK_MV_REQUIRES_GRAD | SK_MV_PARAMETER);

    /* Layer 0: W0 × input + b0 → ReLU */
    sk_model_var* z0  = sk_mv_matmul(arena, model, W0, input, 0);
    sk_model_var* z0b = sk_mv_add(arena, model, z0, b0v, 0);
    sk_model_var* a0  = sk_mv_relu(arena, model, z0b, 0);

    /* Layer 1: W1 × a0 + b1 → ReLU + skip */
    sk_model_var* z1  = sk_mv_matmul(arena, model, W1, a0, 0);
    sk_model_var* z1b = sk_mv_add(arena, model, z1, b1v, 0);
    sk_model_var* z1r = sk_mv_relu(arena, model, z1b, 0);
    sk_model_var* a1  = sk_mv_add(arena, model, a0, z1r, 0);  /* skip connection */

    /* Layer 2: W2 × a1 + b2 → Softmax */
    sk_model_var* z2  = sk_mv_matmul(arena, model, W2, a1, 0);
    sk_model_var* z2b = sk_mv_add(arena, model, z2, b2v, 0);
    sk_model_var* out = sk_mv_softmax(arena, model, z2b, SK_MV_OUTPUT);

    /* Labels + loss */
    sk_model_var* y    = sk_mv_create(arena, model, 10, 1, SK_MV_DESIRED_OUTPUT);
    sk_model_var* cost = sk_mv_cross_entropy(arena, model, y, out, SK_MV_COST);

    (void)cost;
}

/* ── main ─────────────────────────────────────────────────────────── */
int main(void) {
    sk_arena* perm = sk_arena_create(GiB(1), MiB(1));

    sk_matrix* train_images = sk_mat_load(perm, 60000, 784, "train_images.mat");
    sk_matrix* test_images  = sk_mat_load(perm, 10000, 784, "test_images.mat");
    sk_matrix* train_labels = sk_mat_create(perm, 60000, 10);
    sk_matrix* test_labels  = sk_mat_create(perm, 10000, 10);

    {
        sk_matrix* tl = sk_mat_load(perm, 60000, 1, "train_labels.mat");
        sk_matrix* tst = sk_mat_load(perm, 10000, 1, "test_labels.mat");
        for (u32 i = 0; i < 60000; i++)
            train_labels->data[i * 10 + (u32)tl->data[i]] = 1.0f;
        for (u32 i = 0; i < 10000; i++)
            test_labels->data[i * 10 + (u32)tst->data[i]] = 1.0f;
    }

    draw_mnist_digit(test_images->data);
    printf("Label: ");
    for (u32 i = 0; i < 10; i++) printf("%.0f ", test_labels->data[i]);
    printf("\n\n");

    sk_model* model = sk_model_create(perm);
    create_mnist_model(perm, model);
    sk_model_compile(perm, model);

    memcpy(model->input->val->data, test_images->data, sizeof(f32) * 784);
    sk_model_feedforward(model);
    printf("Pre-training: ");
    for (u32 i = 0; i < 10; i++) printf("%.2f ", model->output->val->data[i]);
    printf("\n");

    sk_training_desc desc = {
        .train_images  = train_images,
        .train_labels  = train_labels,
        .test_images   = test_images,
        .test_labels   = test_labels,
        .epochs        = 10,
        .batch_size    = 50,
        .learning_rate = 0.01f,
    };
    sk_model_train(model, &desc);

    memcpy(model->input->val->data, test_images->data, sizeof(f32) * 784);
    sk_model_feedforward(model);
    printf("Post-training: ");
    for (u32 i = 0; i < 10; i++) printf("%.4f ", model->output->val->data[i]);
    printf("\nPredicted: %llu\n", (unsigned long long)sk_mat_argmax(model->output->val));

    sk_arena_destroy(perm);
    return 0;
}
