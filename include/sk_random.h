/* sk_random.h — SnapKitty MLC: PCG32 pseudo-random number generator
 *
 * Algorithm: PCG-XSH-RR (O'Neill 2014). Period 2^64.
 * Stateless reentrant API (_r suffix) + global convenience wrappers.
 *
 * Authors: Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)
 * Source:  SNAPKITTYAGENT9NOVA/MLC
 * License: BSL-1.1 / AGPL-3.0 / MPL-2.0
 */
#ifndef SK_RANDOM_H
#define SK_RANDOM_H

#include "sk_defs.h"

typedef struct {
    u64 state;
    u64 inc;
} sk_prng;

/* ── Seeding ──────────────────────────────────────────────────────── */
void sk_prng_seed_r(sk_prng* rng, u64 initstate, u64 initseq);
void sk_prng_seed(u64 initstate, u64 initseq);

/* ── Generation ───────────────────────────────────────────────────── */
u32  sk_prng_rand_r(sk_prng* rng);
u32  sk_prng_rand(void);

f32  sk_prng_randf_r(sk_prng* rng);  /* [0, 1) */
f32  sk_prng_randf(void);            /* [0, 1) */

/* Compatibility aliases */
#define prng_state      sk_prng
#define prng_seed_r     sk_prng_seed_r
#define prng_seed       sk_prng_seed
#define prng_rand_r     sk_prng_rand_r
#define prng_rand       sk_prng_rand
#define prng_randf_r    sk_prng_randf_r
#define prng_randf      sk_prng_randf

#endif /* SK_RANDOM_H */
