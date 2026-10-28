// ANE boundary transfers (`Step::AnePack` / `Step::AneUnpack` in metal.rs). A surface holds a
// [1, C, 1, S] f32 tensor: element (c, s) at float c * row + s. The graph value holds it at
// s * C + c when `transposed` (a row-major [S, C] activation), else at c * S + s. One 32x32 tile
// per threadgroup of 32x8 threads; transposes go through threadgroup memory so both sides of the
// copy stay coalesced.

struct AneIo {
    uint c;
    uint s;
    uint row;
    uint transposed;
};

kernel void ane_pack(device const float* x [[buffer(0)]],
                     device float* surf [[buffer(1)]],
                     constant AneIo& p [[buffer(2)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float tile[32][33];
    uint c0 = tg.x * 32, s0 = tg.y * 32;
    if (!p.transposed) {
        for (uint j = lid.y; j < 32; j += 8) {
            uint c = c0 + j, s = s0 + lid.x;
            if (c < p.c && s < p.s) surf[(ulong)c * p.row + s] = x[(ulong)c * p.s + s];
        }
        return;
    }
    for (uint j = lid.y; j < 32; j += 8) {
        uint s = s0 + j, c = c0 + lid.x;
        if (c < p.c && s < p.s) tile[j][lid.x] = x[(ulong)s * p.c + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = lid.y; j < 32; j += 8) {
        uint c = c0 + j, s = s0 + lid.x;
        if (c < p.c && s < p.s) surf[(ulong)c * p.row + s] = tile[lid.x][j];
    }
}

kernel void ane_unpack(device float* x [[buffer(0)]],
                       device const float* surf [[buffer(1)]],
                       constant AneIo& p [[buffer(2)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float tile[32][33];
    uint c0 = tg.x * 32, s0 = tg.y * 32;
    if (!p.transposed) {
        for (uint j = lid.y; j < 32; j += 8) {
            uint c = c0 + j, s = s0 + lid.x;
            if (c < p.c && s < p.s) x[(ulong)c * p.s + s] = surf[(ulong)c * p.row + s];
        }
        return;
    }
    for (uint j = lid.y; j < 32; j += 8) {
        uint c = c0 + j, s = s0 + lid.x;
        if (c < p.c && s < p.s) tile[j][lid.x] = surf[(ulong)c * p.row + s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = lid.y; j < 32; j += 8) {
        uint s = s0 + j, c = c0 + lid.x;
        if (c < p.c && s < p.s) x[(ulong)s * p.c + c] = tile[lid.x][j];
    }
}
