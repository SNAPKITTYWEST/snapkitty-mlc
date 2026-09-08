/* sk_defs.h — SnapKitty MLC: type aliases, size macros, utility macros
 *
 * Authors: Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)
 * Source:  SNAPKITTYAGENT9NOVA/MLC
 * License: BSL-1.1 / AGPL-3.0 / MPL-2.0
 */
#ifndef SK_DEFS_H
#define SK_DEFS_H

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

/* ── Integer type aliases ─────────────────────────────────────────── */
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef i8  b8;
typedef i32 b32;

typedef float f32;

/* ── Size helpers ─────────────────────────────────────────────────── */
#define KiB(n) ((u64)(n) << 10)
#define MiB(n) ((u64)(n) << 20)
#define GiB(n) ((u64)(n) << 30)

/* ── Utility macros ───────────────────────────────────────────────── */
#define SK_MIN(a, b)           (((a) < (b)) ? (a) : (b))
#define SK_MAX(a, b)           (((a) > (b)) ? (a) : (b))
#define SK_ALIGN_UP_POW2(n, p) (((u64)(n) + ((u64)(p) - 1)) & (~((u64)(p) - 1)))

/* Compatibility aliases (original names) */
#define MIN SK_MIN
#define MAX SK_MAX
#define ALIGN_UP_POW2 SK_ALIGN_UP_POW2

#endif /* SK_DEFS_H */
