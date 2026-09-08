/* sk_arena.h — SnapKitty MLC: virtual-memory arena allocator
 *
 * API:
 *   sk_arena_create / sk_arena_destroy
 *   sk_arena_push   / sk_arena_pop / sk_arena_pop_to / sk_arena_clear
 *   sk_arena_temp_begin / sk_arena_temp_end
 *   sk_arena_scratch_get / sk_arena_scratch_release   (2 thread-local scratch pools)
 *
 * Platform backends: Win32 (VirtualAlloc) and POSIX (mmap/mprotect).
 *
 * Authors: Ahmad Ali Parr, Jessica L. Williams (SNAPKITTYWEST)
 * Source:  SNAPKITTYAGENT9NOVA/MLC
 * License: BSL-1.1 / AGPL-3.0 / MPL-2.0
 */
#ifndef SK_ARENA_H
#define SK_ARENA_H

#include "sk_defs.h"

/* ── Arena header lives at the start of the reserved region ──────── */
#define SK_ARENA_BASE_POS (sizeof(sk_arena))
#define SK_ARENA_ALIGN    (sizeof(void*))

typedef struct {
    u64 reserve_size;
    u64 commit_size;
    u64 pos;
    u64 commit_pos;
} sk_arena;

typedef struct {
    sk_arena* arena;
    u64       start_pos;
} sk_arena_temp;

/* ── Lifecycle ────────────────────────────────────────────────────── */
sk_arena*      sk_arena_create(u64 reserve_size, u64 commit_size);
void           sk_arena_destroy(sk_arena* arena);

/* ── Allocation ───────────────────────────────────────────────────── */
void*          sk_arena_push(sk_arena* arena, u64 size, b32 non_zero);
void           sk_arena_pop(sk_arena* arena, u64 size);
void           sk_arena_pop_to(sk_arena* arena, u64 pos);
void           sk_arena_clear(sk_arena* arena);

/* ── Temporary sub-arenas ─────────────────────────────────────────── */
sk_arena_temp  sk_arena_temp_begin(sk_arena* arena);
void           sk_arena_temp_end(sk_arena_temp temp);

/* ── Thread-local scratch pools (2 arenas, conflict-aware) ────────── */
sk_arena_temp  sk_arena_scratch_get(sk_arena** conflicts, u32 num_conflicts);
void           sk_arena_scratch_release(sk_arena_temp scratch);

/* ── Push macros ──────────────────────────────────────────────────── */
#define SK_PUSH_STRUCT(arena, T)       ((T*)sk_arena_push((arena), sizeof(T), false))
#define SK_PUSH_STRUCT_NZ(arena, T)    ((T*)sk_arena_push((arena), sizeof(T), true))
#define SK_PUSH_ARRAY(arena, T, n)     ((T*)sk_arena_push((arena), sizeof(T) * (n), false))
#define SK_PUSH_ARRAY_NZ(arena, T, n)  ((T*)sk_arena_push((arena), sizeof(T) * (n), true))

/* Compatibility aliases (original names) */
#define PUSH_STRUCT    SK_PUSH_STRUCT
#define PUSH_STRUCT_NZ SK_PUSH_STRUCT_NZ
#define PUSH_ARRAY     SK_PUSH_ARRAY
#define PUSH_ARRAY_NZ  SK_PUSH_ARRAY_NZ
#define arena_create          sk_arena_create
#define arena_destroy         sk_arena_destroy
#define arena_push            sk_arena_push
#define arena_pop             sk_arena_pop
#define arena_pop_to          sk_arena_pop_to
#define arena_clear           sk_arena_clear
#define arena_temp_begin      sk_arena_temp_begin
#define arena_temp_end        sk_arena_temp_end
#define arena_scratch_get     sk_arena_scratch_get
#define arena_scratch_release sk_arena_scratch_release
#define mem_arena      sk_arena
#define mem_arena_temp sk_arena_temp
#define ARENA_BASE_POS SK_ARENA_BASE_POS
#define ARENA_ALIGN    SK_ARENA_ALIGN

/* ── Platform primitives (implemented per-OS in sk_arena.c) ─────── */
u32  plat_get_pagesize(void);
void* plat_mem_reserve(u64 size);
b32  plat_mem_commit(void* ptr, u64 size);
b32  plat_mem_decommit(void* ptr, u64 size);
b32  plat_mem_release(void* ptr, u64 size);

#endif /* SK_ARENA_H */
