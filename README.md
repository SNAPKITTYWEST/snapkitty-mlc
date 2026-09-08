# snapkitty-mlc

**SnapKitty Machine Learning in C** — minimal autograd library, arena allocator, PCG32 PRNG, MNIST example.

Authors: Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)  
Source: [SNAPKITTYAGENT9NOVA/MLC](https://github.com/SNAPKITTYAGENT9NOVA/MLC)  
License: BSL-1.1 / AGPL-3.0 / MPL-2.0

---

## Library layout

```
snapkitty-mlc/
├── include/
│   ├── sk_defs.h       Type aliases (i8/u8/f32…), KiB/MiB/GiB, MIN/MAX
│   ├── sk_arena.h      Virtual-memory arena allocator (Win32 + POSIX)
│   ├── sk_random.h     PCG32 PRNG (O'Neill 2014) — reentrant + global
│   ├── sk_matrix.h     Row-major f32 matrix: create/fill/matmul/relu/softmax/xent
│   └── sk_model.h      Computation graph, autograd, mini-batch SGD
├── src/
│   ├── sk_arena.c      Arena implementation (VirtualAlloc / mmap)
│   ├── sk_random.c     PCG32 implementation
│   ├── sk_matrix.c     All matrix ops + gradient accumulation
│   └── sk_model.c      Graph build, topological sort, forward/backward, train loop
├── examples/
│   ├── mnist.c         784→16 ReLU → 16+skip ReLU → 10 Softmax (cross-entropy)
│   └── data_convert.py MNIST → binary .mat (requires tensorflow-datasets)
└── CMakeLists.txt
```

---

## Build

```bash
cmake -S . -B build
cmake --build build
# → build/libsk_mlc.a  build/mnist
```

Requires: C11 compiler (GCC / Clang / MSVC), CMake ≥ 3.16, no other dependencies.

---

## Run MNIST

```bash
# 1. Generate data (once)
pip install tensorflow-datasets numpy
python examples/data_convert.py
# → train_images.mat  train_labels.mat  test_images.mat  test_labels.mat

# 2. Train
./build/mnist
```

Expected: ~97% test accuracy after 10 epochs (784→16→16→10 with skip, SGD lr=0.01, batch=50).

---

## API reference

### Arena allocator (`sk_arena.h`)

```c
sk_arena* sk_arena_create(u64 reserve_size, u64 commit_size);
void      sk_arena_destroy(sk_arena* arena);
void*     sk_arena_push(sk_arena* arena, u64 size, b32 non_zero);
void      sk_arena_clear(sk_arena* arena);

sk_arena_temp sk_arena_temp_begin(sk_arena* arena);
void          sk_arena_temp_end(sk_arena_temp temp);

sk_arena_temp sk_arena_scratch_get(sk_arena** conflicts, u32 n);
void          sk_arena_scratch_release(sk_arena_temp scratch);

SK_PUSH_STRUCT(arena, T)     // allocate one T, zeroed
SK_PUSH_ARRAY(arena, T, n)   // allocate n × T, zeroed
```

### PRNG (`sk_random.h`)

```c
void sk_prng_seed(u64 initstate, u64 initseq);
u32  sk_prng_rand(void);       // [0, 2^32)
f32  sk_prng_randf(void);      // [0, 1)

// Reentrant variants:
void sk_prng_seed_r(sk_prng* rng, u64 s, u64 seq);
u32  sk_prng_rand_r(sk_prng* rng);
f32  sk_prng_randf_r(sk_prng* rng);
```

### Matrix (`sk_matrix.h`)

```c
sk_matrix* sk_mat_create(sk_arena*, u32 rows, u32 cols);
sk_matrix* sk_mat_load(sk_arena*, u32 rows, u32 cols, const char* path);

void sk_mat_fill_rand(sk_matrix* mat, f32 lo, f32 hi);  // Xavier init
void sk_mat_clear(sk_matrix* mat);

b32  sk_mat_mul(out, a, b, zero_out, transpose_a, transpose_b);
b32  sk_mat_relu(out, in);
b32  sk_mat_softmax(out, in);
b32  sk_mat_cross_entropy(out, p, q);

// Gradient accumulation (adds into out):
b32  sk_mat_relu_add_grad(out, in, grad);
b32  sk_mat_softmax_add_grad(out, softmax_out, grad);
b32  sk_mat_cross_entropy_add_grad(p_grad, q_grad, p, q, grad);
```

### Model / autograd (`sk_model.h`)

```c
sk_model* sk_model_create(sk_arena* arena);

// Variable constructors (build computation graph):
sk_model_var* sk_mv_create(arena, model, rows, cols, flags);
sk_model_var* sk_mv_matmul(arena, model, a, b, flags);
sk_model_var* sk_mv_add(arena, model, a, b, flags);
sk_model_var* sk_mv_relu(arena, model, input, flags);
sk_model_var* sk_mv_softmax(arena, model, input, flags);
sk_model_var* sk_mv_cross_entropy(arena, model, p, q, flags);

// Compile + run:
void sk_model_compile(arena, model);     // topological sort
void sk_model_feedforward(model);        // forward pass
void sk_model_train(model, &desc);       // SGD training loop

// Flags:
SK_MV_REQUIRES_GRAD | SK_MV_PARAMETER | SK_MV_INPUT | SK_MV_OUTPUT
SK_MV_DESIRED_OUTPUT | SK_MV_COST
```

---

## Design notes

- **No malloc** in the hot path — all allocation through the arena.
- **No external dependencies** — libc only (`stdio`, `string`, `math`).
- **PCG32** — better statistical properties than `rand()`, reproducible with seed.
- **Topological sort** is a single iterative DFS; program arrays are pre-built so forward/backward are simple loops.
- **Gradient accumulation** — `_add_grad` functions accumulate into existing grad arrays; caller clears before each batch.
- All `sk_` prefixes have **compatibility aliases** (`mat_create` etc.) so the original MLC code compiles unchanged.

---

## License

BSL-1.1 / AGPL-3.0 / MPL-2.0  
SnapKitty West / SNAPKITTYWEST — Evidence or Silence — 2026
