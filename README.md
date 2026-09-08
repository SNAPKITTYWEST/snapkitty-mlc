[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](https://opensource.org/licenses/BSL-1.0)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-purple.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL--2.0-orange.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
[![C11](https://img.shields.io/badge/C-C11-00599C.svg)](https://en.cppreference.com/w/c/11)
[![CMake](https://img.shields.io/badge/CMake-%E2%89%A53.16-064F8C.svg)](https://cmake.org/)
[![FlashAttention](https://img.shields.io/badge/FlashAttention-Dao%202022-green.svg)](https://arxiv.org/abs/2205.14135)
[![Evidence or Silence](https://img.shields.io/badge/Protocol-Evidence%20or%20Silence-black.svg)](#)

# snapkitty-mlc

**SnapKitty Machine Learning in C** — 機器學習 C 語言核心庫 (مكتبة التعلم الآلي الأساسية بلغة C)

Minimal autograd library with arena allocator, PCG32 PRNG, and MNIST example. Extracted from [SNAPKITTYAGENT9NOVA/MLC](https://github.com/SNAPKITTYAGENT9NOVA/MLC) and extended with FlashAttention (Dao et al. 2022).

Authors: Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)

---

## Modules

| Module | Description | 描述 (الوصف) |
|--------|-------------|--------------|
| `sk_arena` | Virtual-memory arena allocator — Win32 VirtualAlloc + Linux mmap; 2 thread-local scratch pools | 虛擬記憶體分配器 (مخصص الذاكرة الافتراضية) |
| `sk_random` | PCG32 (O'Neill 2014) — reentrant `_r` variants, better statistical properties than `rand()` | 偽隨機數生成器 (مولد الأرقام العشوائية الزائفة) |
| `sk_matrix` | Row-major f32 matrix — all 4 matmul transpose variants, ReLU/Softmax/CrossEntropy + gradient accumulation | 矩陣運算 + 梯度累積 (عمليات المصفوفات + تراكم التدرج) |
| `sk_model` | Computation graph — iterative DFS topological sort, reverse-mode autograd, mini-batch SGD with Fisher-Yates shuffle | 計算圖 + 自動微分 (الرسم البياني الحسابي + التفاضل التلقائي) |

---

## Repository Structure

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
│   ├── mnist.c                    784→16 ReLU → 16+skip ReLU → 10 Softmax
│   ├── flash_attention_golden.py  FlashAttention Alg 1 (Dao et al. 2022)
│   └── data_convert.py            MNIST → binary .mat
└── CMakeLists.txt
```

---

## Build

```bash
cmake -S . -B build
cmake --build build
# → build/libsk_mlc.a  build/mnist
```

**Requirements:** C11 compiler (GCC / Clang / MSVC), CMake ≥ 3.16. No external dependencies.

---

## MNIST Example

```bash
# 1. Generate data (once)
pip install tensorflow-datasets numpy
python examples/data_convert.py

# 2. Train
./build/mnist
```

Architecture: `784 → 16 (ReLU) → 16+skip (ReLU) → 10 (Softmax+CrossEntropy)`
Expected: ~97% test accuracy after 10 epochs (SGD lr=0.01, batch=50).

---

## FlashAttention Verification

```bash
python3 examples/flash_attention_golden.py
# ALL TESTS PASSED — max error 2×10⁻¹²
```

All 4 test configurations pass against Algorithm 1 from Dao et al. 2022.

---

## API Reference

### Arena Allocator (`sk_arena.h`)

```c
sk_arena* sk_arena_create(u64 reserve_size, u64 commit_size);
void      sk_arena_destroy(sk_arena* arena);
void*     sk_arena_push(sk_arena* arena, u64 size, b32 non_zero);
void      sk_arena_clear(sk_arena* arena);

SK_PUSH_STRUCT(arena, T)     // allocate one T, zeroed
SK_PUSH_ARRAY(arena, T, n)   // allocate n × T, zeroed
```

### PRNG (`sk_random.h`)

```c
void sk_prng_seed(u64 initstate, u64 initseq);
u32  sk_prng_rand(void);       // [0, 2^32)
f32  sk_prng_randf(void);      // [0, 1)

// Reentrant:
void sk_prng_seed_r(sk_prng* rng, u64 s, u64 seq);
u32  sk_prng_rand_r(sk_prng* rng);
f32  sk_prng_randf_r(sk_prng* rng);
```

### Matrix (`sk_matrix.h`)

```c
sk_matrix* sk_mat_create(sk_arena*, u32 rows, u32 cols);
b32  sk_mat_mul(out, a, b, zero_out, transpose_a, transpose_b);
b32  sk_mat_relu(out, in);
b32  sk_mat_softmax(out, in);
b32  sk_mat_cross_entropy(out, p, q);
```

### Model / Autograd (`sk_model.h`)

```c
sk_model* sk_model_create(sk_arena* arena);
sk_model_var* sk_mv_matmul(arena, model, a, b, flags);
sk_model_var* sk_mv_relu(arena, model, input, flags);
sk_model_var* sk_mv_softmax(arena, model, input, flags);
sk_model_var* sk_mv_cross_entropy(arena, model, p, q, flags);
void sk_model_compile(arena, model);     // topological sort
void sk_model_train(model, &desc);       // SGD training loop
```

---

## Design Principles

- **No malloc** in the hot path — all allocation through the arena
- **No external dependencies** — libc only (`stdio`, `string`, `math`)
- **PCG32** — better statistical properties than `rand()`, reproducible with seed
- **Iterative DFS** topological sort — forward/backward are simple array loops
- **Gradient accumulation** — `_add_grad` functions accumulate; caller clears per batch

---

## License

This project is released under a **trilicense** model. You may choose any one of the following:

| License | SPDX | Link |
|---------|------|------|
| Boost Software License 1.0 | BSL-1.1 | [LICENSE-BSL](https://opensource.org/licenses/BSL-1.0) |
| GNU Affero General Public License v3 | AGPL-3.0 | [LICENSE-AGPL](https://www.gnu.org/licenses/agpl-3.0) |
| Mozilla Public License 2.0 | MPL-2.0 | [LICENSE-MPL](https://www.mozilla.org/en-US/MPL/2.0/) |

Unauthorized cloud SaaS redistribution without source disclosure is prohibited under all three licenses.

---

SnapKitty West / SNAPKITTYWEST — Evidence or Silence — 2026
