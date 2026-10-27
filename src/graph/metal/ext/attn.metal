// Kernels for the `attn` lowering group (see attn.rs). Shares the preamble of metal_kernels.metal.

// ---- Flash-style attention (`Op::Attention`) ---------------------------------------------------
// One threadgroup per (head, block of ATTNF_NSG query rows), one simdgroup per query row. The block's
// K/V rows are staged through threadgroup memory in tiles shared by its simdgroups; every simdgroup
// runs an online softmax over its own visible key range. Lane `l` owns head dims l, l+32, ...
// (`NJ` per lane, so head_dim <= 32 * NJ). Handles grouped kv heads, causal / windowed masks, soft
// cap and an additive bias; a query with no visible key (or only -inf bias) writes zeros.
// ponytail: scalar-FMA dot products, ~0.3 TFLOP/s measured (4096x4096, 24 heads, hd 128, M-series);
// a query-block simdgroup_matrix tile should be several times faster for long DiT/ViT sequences. A
// short query block (nq < ATTNF_NSG) leaves simdgroups idle and does not split keys across them.
constant uint ATTNF_NSG = 8;     // simdgroups (= query rows) per threadgroup; attn.rs `QB`
constant uint ATTNF_TILE = 2048; // floats per staged K or V tile (K + V = 16 KiB)

struct AttnFlashParams {
    uint nq;
    uint nk;
    uint heads;
    uint kv_heads;
    uint head_dim;
    uint flags;     // bit 0 causal, bit 1 window, bit 2 soft cap
    uint window;
    uint bias_mode; // 0 none, 1 [nk], 2 [nq, nk], 3 [heads, nq, nk]
    float scale;
    float soft_cap;
};

// Keys [lo, hi) visible to the query at `pos` (mirrors `cpu::attention_key_range`). Both bounds
// are non-decreasing in `pos`, which the threadgroup-wide tile range below relies on.
static inline void attn_key_range(uint pos, constant AttnFlashParams& p, thread uint& lo, thread uint& hi) {
    const bool causal = (p.flags & 1u) != 0;
    const bool win = (p.flags & 2u) != 0;
    const uint half_w = p.window / 2;
    if (causal) {
        hi = min(p.nk, pos + 1);
        lo = (win && pos + 1 > p.window) ? pos + 1 - p.window : 0;
    } else {
        hi = win ? min(p.nk, pos + half_w + 1) : p.nk;
        lo = (win && pos > half_w) ? pos - half_w : 0;
    }
}

template <uint NJ>
static inline void attn_flash(device const float* q, device const float* k, device const float* v,
                              device const float* bias, device float* out,
                              constant AttnFlashParams& p,
                              threadgroup float* tk, threadgroup float* tv,
                              uint2 tgp, uint sg, uint lane) {
    const uint hd = p.head_dim;
    const uint head = tgp.y;
    const uint kvh = head / (p.heads / p.kv_heads);
    const uint q0 = tgp.x * ATTNF_NSG;
    const uint qi = q0 + sg;
    const bool active = qi < p.nq;
    const ulong qstride = (ulong)p.heads * hd;
    const ulong kvstride = (ulong)p.kv_heads * hd;
    const bool has_cap = (p.flags & 4u) != 0;

    // Threadgroup-wide key range: the first row has the smallest `lo`, the last row the largest `hi`.
    uint tg_lo = 0, tg_hi = 0, lo = 0, hi = 0, unused = 0;
    attn_key_range(q0, p, tg_lo, unused);
    attn_key_range(min(q0 + ATTNF_NSG, p.nq) - 1, p, unused, tg_hi);
    if (active) attn_key_range(qi, p, lo, hi);

    float qr[NJ];
    float acc[NJ];
    for (uint j = 0; j < NJ; ++j) {
        const uint d = lane + 32 * j;
        qr[j] = (active && d < hd) ? q[(ulong)qi * qstride + (ulong)head * hd + d] : 0.0f;
        acc[j] = 0.0f;
    }
    ulong brow = 0;
    if (p.bias_mode == 2) brow = (ulong)qi * p.nk;
    else if (p.bias_mode == 3) brow = ((ulong)head * p.nq + qi) * p.nk;

    float m = -FLT_MAX;
    float l = 0.0f;
    const uint kt = ATTNF_TILE / hd;
    for (uint t0 = tg_lo; t0 < tg_hi; t0 += kt) {
        const uint n = min(kt, tg_hi - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup); // the previous tile is consumed
        for (uint r = sg; r < n; r += ATTNF_NSG) {
            device const float* kr = k + (ulong)(t0 + r) * kvstride + (ulong)kvh * hd;
            device const float* vr = v + (ulong)(t0 + r) * kvstride + (ulong)kvh * hd;
            for (uint d = lane; d < hd; d += 32) {
                tk[r * hd + d] = kr[d];
                tv[r * hd + d] = vr[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!active) continue;
        const uint a = max(t0, lo);
        const uint b = min(t0 + n, hi);
        // Four keys per step: independent dot/simd_sum chains, one softmax rescale per step.
        for (uint t = a; t < b; t += 4) {
            float s[4];
            for (uint u = 0; u < 4; ++u) {
                float x = 0.0f;
                if (t + u < b) {
                    threadgroup const float* kr = tk + (t + u - t0) * hd;
                    for (uint j = 0; j < NJ; ++j) {
                        const uint d = lane + 32 * j;
                        if (d < hd) x += qr[j] * kr[d];
                    }
                }
                s[u] = x;
            }
            float m_new = m;
            for (uint u = 0; u < 4; ++u) {
                float x = simd_sum(s[u]) * p.scale;
                if (t + u < b) {
                    if (has_cap) x = p.soft_cap * tanh(x / p.soft_cap);
                    if (p.bias_mode != 0) x += bias[brow + t + u];
                } else {
                    x = -INFINITY;
                }
                s[u] = x;
                m_new = max(m_new, x);
            }
            const float corr = exp(m - m_new);
            l *= corr;
            for (uint j = 0; j < NJ; ++j) acc[j] *= corr;
            for (uint u = 0; u < 4; ++u) {
                const float e = exp(s[u] - m_new); // 0 for a -inf score
                l += e;
                if (t + u < b) {
                    threadgroup const float* vr = tv + (t + u - t0) * hd;
                    for (uint j = 0; j < NJ; ++j) {
                        const uint d = lane + 32 * j;
                        if (d < hd) acc[j] += e * vr[d];
                    }
                }
            }
            m = m_new;
        }
    }
    if (active) {
        device float* o = out + (ulong)qi * qstride + (ulong)head * hd;
        for (uint j = 0; j < NJ; ++j) {
            const uint d = lane + 32 * j;
            if (d < hd) o[d] = l > 0.0f ? acc[j] / l : 0.0f;
        }
    }
}

#define ATTNF_KERNEL(NJ)                                                                         \
kernel void attn_flash_nj##NJ(device const float* q [[buffer(0)]],                               \
                              device const float* k [[buffer(1)]],                               \
                              device const float* v [[buffer(2)]],                               \
                              device const float* bias [[buffer(3)]],                            \
                              device float* out [[buffer(4)]],                                   \
                              constant AttnFlashParams& p [[buffer(5)]],                         \
                              uint2 tgp [[threadgroup_position_in_grid]],                        \
                              uint sg [[simdgroup_index_in_threadgroup]],                        \
                              uint lane [[thread_index_in_simdgroup]]) {                         \
    threadgroup float tk[ATTNF_TILE];                                                            \
    threadgroup float tv[ATTNF_TILE];                                                            \
    attn_flash<NJ>(q, k, v, bias, out, p, tk, tv, tgp, sg, lane);                               \
}

ATTNF_KERNEL(2)
ATTNF_KERNEL(3)
ATTNF_KERNEL(4)
ATTNF_KERNEL(8)
ATTNF_KERNEL(16)

// Few-query (`nq < 8`) split-key attention: one threadgroup (ATTNF_NSG = 8 simdgroups) per
// `(query_row, head)`. Simdgroup `sg` processes every 8th key chunk in `[lo, hi)` with its own
// online-softmax `(m, l, acc[NJ])`, then the 8 partials are merged through threadgroup memory.
template <uint NJ>
static inline void attn_flash_split(device const float* q, device const float* k, device const float* v,
                                    device const float* bias, device float* out,
                                    constant AttnFlashParams& p,
                                    threadgroup float* sm_m, threadgroup float* sm_l,
                                    threadgroup float* sm_acc,
                                    uint2 tgp, uint sg, uint lane) {
    const uint hd = p.head_dim;
    const uint head = tgp.y;
    const uint kvh = head / (p.heads / p.kv_heads);
    const uint qi = tgp.x;
    if (qi >= p.nq) return;
    const ulong qstride = (ulong)p.heads * hd;
    const ulong kvstride = (ulong)p.kv_heads * hd;
    const bool has_cap = (p.flags & 4u) != 0;

    uint lo = 0, hi = 0;
    attn_key_range(qi, p, lo, hi);

    float qr[NJ];
    float acc[NJ];
    for (uint j = 0; j < NJ; ++j) {
        const uint d = lane + 32 * j;
        qr[j] = (d < hd) ? q[(ulong)qi * qstride + (ulong)head * hd + d] : 0.0f;
        acc[j] = 0.0f;
    }
    ulong brow = 0;
    if (p.bias_mode == 2) brow = (ulong)qi * p.nk;
    else if (p.bias_mode == 3) brow = ((ulong)head * p.nq + qi) * p.nk;

    float m = -FLT_MAX;
    float l = 0.0f;
    for (uint t = lo + sg * 4; t < hi; t += ATTNF_NSG * 4) {
        float s[4];
        for (uint u = 0; u < 4; ++u) {
            float x = 0.0f;
            if (t + u < hi) {
                device const float* kr = k + (ulong)(t + u) * kvstride + (ulong)kvh * hd;
                for (uint j = 0; j < NJ; ++j) {
                    const uint d = lane + 32 * j;
                    if (d < hd) x += qr[j] * kr[d];
                }
            }
            s[u] = x;
        }
        float m_new = m;
        for (uint u = 0; u < 4; ++u) {
            float x = simd_sum(s[u]) * p.scale;
            if (t + u < hi) {
                if (has_cap) x = p.soft_cap * tanh(x / p.soft_cap);
                if (p.bias_mode != 0) x += bias[brow + t + u];
            } else {
                x = -INFINITY;
            }
            s[u] = x;
            m_new = max(m_new, x);
        }
        const float corr = exp(m - m_new);
        l *= corr;
        for (uint j = 0; j < NJ; ++j) acc[j] *= corr;
        for (uint u = 0; u < 4; ++u) {
            const float e = exp(s[u] - m_new);
            l += e;
            if (t + u < hi) {
                device const float* vr = v + (ulong)(t + u) * kvstride + (ulong)kvh * hd;
                for (uint j = 0; j < NJ; ++j) {
                    const uint d = lane + 32 * j;
                    if (d < hd) acc[j] += e * vr[d];
                }
            }
        }
        m = m_new;
    }

    if (lane == 0) {
        sm_m[sg] = m;
        sm_l[sg] = l;
    }
    for (uint j = 0; j < NJ; ++j) {
        const uint d = lane + 32 * j;
        if (d < hd) sm_acc[sg * (32 * NJ) + d] = acc[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float m_tot = -FLT_MAX;
        for (uint s_idx = 0; s_idx < ATTNF_NSG; ++s_idx) {
            if (sm_l[s_idx] > 0.0f) m_tot = max(m_tot, sm_m[s_idx]);
        }
        float l_tot = 0.0f;
        float w_sg[ATTNF_NSG];
        for (uint s_idx = 0; s_idx < ATTNF_NSG; ++s_idx) {
            float w = (sm_l[s_idx] > 0.0f) ? exp(sm_m[s_idx] - m_tot) : 0.0f;
            w_sg[s_idx] = w;
            l_tot += sm_l[s_idx] * w;
        }
        device float* o = out + (ulong)qi * qstride + (ulong)head * hd;
        const float inv = l_tot > 0.0f ? (1.0f / l_tot) : 0.0f;
        for (uint j = 0; j < NJ; ++j) {
            const uint d = lane + 32 * j;
            if (d < hd) {
                float sum = 0.0f;
                for (uint s_idx = 0; s_idx < ATTNF_NSG; ++s_idx) {
                    sum += sm_acc[s_idx * (32 * NJ) + d] * w_sg[s_idx];
                }
                o[d] = sum * inv;
            }
        }
    }
}

#define ATTNF_SPLIT_KERNEL(NJ)                                                                   \
kernel void attn_flash_split_nj##NJ(device const float* q [[buffer(0)]],                         \
                                    device const float* k [[buffer(1)]],                         \
                                    device const float* v [[buffer(2)]],                         \
                                    device const float* bias [[buffer(3)]],                      \
                                    device float* out [[buffer(4)]],                             \
                                    constant AttnFlashParams& p [[buffer(5)]],                   \
                                    uint2 tgp [[threadgroup_position_in_grid]],                  \
                                    uint sg [[simdgroup_index_in_threadgroup]],                  \
                                    uint lane [[thread_index_in_simdgroup]]) {                   \
    threadgroup float sm_m[ATTNF_NSG];                                                           \
    threadgroup float sm_l[ATTNF_NSG];                                                           \
    threadgroup float sm_acc[ATTNF_NSG * 32 * NJ];                                               \
    attn_flash_split<NJ>(q, k, v, bias, out, p, sm_m, sm_l, sm_acc, tgp, sg, lane);              \
}

ATTNF_SPLIT_KERNEL(2)
ATTNF_SPLIT_KERNEL(3)
ATTNF_SPLIT_KERNEL(4)
ATTNF_SPLIT_KERNEL(8)
ATTNF_SPLIT_KERNEL(16)

#if defined(GIBBS_MPP)
// Tiled flash attention on the matrix units (`mpp::tensor_ops::matmul2d`): one threadgroup (4
// simdgroups = 128 threads) per (query block of QB rows, query head). For f32 K/V (`Op::Attention`
// and f32 `Step::CachedAttention`), Q, K and V are read directly as strided `device float` tensors
// with no threadgroup staging or half truncation; only the `[QB, C]` tile scores/weights `ts` live
// in threadgroup memory (4 KiB), allowing QB = 32, C = 32 at every HD up to 512.
template <uint HD, uint QB>
static inline void flash_attn_mm_f32(device const float* q, device const float* k, device const float* v,
                                     device const float* bias, device float* out,
                                     uint nq, uint nk, uint heads, uint kv_heads, uint head_dim,
                                     ulong q_stride, ulong kv_stride, float scale,
                                     bool has_cap, float soft_cap, uint bias_mode,
                                     uint row_lo_lane, uint row_hi_lane,
                                     threadgroup float* ts,
                                     threadgroup float* tg_corr,
                                     threadgroup uint* tg_row_lo, threadgroup uint* tg_row_hi,
                                     threadgroup uint* tg_bounds,
                                     uint2 tgp, uint sg, uint lane,
                                     uint paged = 0, uint ring = 0) {
    constexpr uint C = 32;
    constexpr uint ROWS_PER_SG = QB / 4;
    const uint hd = head_dim;
    const uint head = tgp.y;
    const uint kvh = head / (heads / kv_heads);
    const uint q0 = tgp.x * QB;
    if (q0 >= nq) return;
    const uint q_ext = min(QB, nq - q0);
    const ulong q_head_off = (ulong)head * hd;
    const ulong kv_head_off = (ulong)kvh * hd;

    if (sg == 0) {
        if (lane < QB) {
            tg_row_lo[lane] = row_lo_lane;
            tg_row_hi[lane] = row_hi_lane;
        }
        const uint min_lo = simd_min(row_lo_lane);
        const uint max_hi = simd_max(row_hi_lane);
        if (lane == 0) {
            tg_bounds[0] = min_lo;
            tg_bounds[1] = max_hi;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Paged K/V (`kv_row`): key tiles start on a multiple of C, so none straddles a 64-row page
    // (keys below a row's `lo` are masked).
    const uint tg_lo = paged ? (tg_bounds[0] & ~(C - 1)) : tg_bounds[0];
    const uint tg_hi = tg_bounds[1];

    float m_row[ROWS_PER_SG];
    float l_row[ROWS_PER_SG];
    #pragma unroll
    for (uint rh = 0; rh < ROWS_PER_SG; ++rh) {
        m_row[rh] = -INFINITY;
        l_row[rh] = 0.0f;
    }

    auto tq_v = tensor((device float*)(q + (ulong)q0 * q_stride + q_head_off),
                       dextents<int32_t, 2>(int(hd), int(q_ext)),
                       array<int, 2>({1, int(q_stride)}));
    constexpr uint PV_N = (HD > 256) ? 256 : HD;
    constexpr uint PV_TILES = HD / PV_N;
    auto tp_proto = tensor(ts, dextents<int32_t, 2>(int(C), int(QB)), array<int, 2>({1, int(C)}));
    auto tv_proto = tensor((device float*)v, dextents<int32_t, 2>(int(PV_N), int(C)), array<int, 2>({1, int(kv_stride)}));

    constexpr auto desc_qk = matmul2d_descriptor(
        int(QB), int(C), static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc_qk, execution_simdgroups<4>> op_qk;
    constexpr auto desc_pv = matmul2d_descriptor(
        int(QB), int(PV_N), static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroups<4>> op_pv;

    auto co0 = op_pv.template get_destination_cooperative_tensor<decltype(tp_proto), decltype(tv_proto), float>();
    auto co1 = op_pv.template get_destination_cooperative_tensor<decltype(tp_proto), decltype(tv_proto), float>();
    #pragma unroll
    for (uint16_t i = 0; i < co0.get_capacity(); ++i) {
        if (co0.is_valid_element(i)) co0[i] = 0.0f;
        if (PV_TILES > 1 && co1.is_valid_element(i)) co1[i] = 0.0f;
    }

    for (uint t0 = tg_lo; t0 < tg_hi; t0 += C) {
        const uint n_keys = min(C, tg_hi - t0);
        auto tk_v = tensor((device float*)(kv_row(k, paged, ring, t0, kv_stride) + kv_head_off),
                           dextents<int32_t, 2>(int(hd), int(n_keys)),
                           array<int, 2>({1, int(kv_stride)}));
        auto cs = op_qk.template get_destination_cooperative_tensor<decltype(tq_v), decltype(tk_v), float>();
        #pragma unroll
        for (uint16_t i = 0; i < cs.get_capacity(); ++i) {
            if (cs.is_valid_element(i)) cs[i] = 0.0f;
        }
        op_qk.run(tq_v, tk_v, cs);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint16_t i = 0; i < cs.get_capacity(); ++i) {
            if (cs.is_valid_element(i)) {
                auto idx = cs.get_multidimensional_index(i);
                const uint col = uint(idx[0]);
                const uint row = uint(idx[1]);
                const uint qi = q0 + row;
                const uint ki = t0 + col;
                float s = -INFINITY;
                if (row < q_ext && col < n_keys && ki >= tg_row_lo[row] && ki < tg_row_hi[row]) {
                    s = cs[i] * scale;
                    if (has_cap) s = soft_cap * tanh(s / soft_cap);
                    if (bias_mode != 0) {
                        const ulong brow = (bias_mode == 2) ? (ulong)qi * nk
                                         : (bias_mode == 3) ? ((ulong)head * nq + qi) * nk
                                         : 0ul;
                        s += bias[brow + ki];
                    }
                }
                if (row < QB && col < C) ts[row * C + col] = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint rh = 0; rh < ROWS_PER_SG; ++rh) {
            const uint r = sg + rh * 4;
            const float s = (r < q_ext && lane < n_keys) ? ts[r * C + lane] : -INFINITY;
            const float m_tile = simd_max(s);
            if (m_tile > -INFINITY) {
                const float m_old = m_row[rh];
                const float m_new = max(m_old, m_tile);
                const float corr = (m_old > -INFINITY) ? exp(m_old - m_new) : 0.0f;
                const float e = (s > -INFINITY) ? exp(s - m_new) : 0.0f;
                l_row[rh] = l_row[rh] * corr + simd_sum(e);
                m_row[rh] = m_new;
                ts[r * C + lane] = e;
                if (lane == 0) tg_corr[r] = corr;
            } else {
                ts[r * C + lane] = 0.0f;
                if (lane == 0) tg_corr[r] = 1.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint16_t i = 0; i < co0.get_capacity(); ++i) {
            if (co0.is_valid_element(i)) {
                auto idx = co0.get_multidimensional_index(i);
                const uint row = uint(idx[1]);
                if (row < QB) {
                    co0[i] *= tg_corr[row];
                    if (PV_TILES > 1) co1[i] *= tg_corr[row];
                }
            }
        }
        auto tp_v = tensor(ts, dextents<int32_t, 2>(int(n_keys), int(q_ext)), array<int, 2>({1, int(C)}));
        const uint hd0 = min(PV_N, hd);
        auto tv_v0 = tensor((device float*)(kv_row(v, paged, ring, t0, kv_stride) + kv_head_off),
                            dextents<int32_t, 2>(int(hd0), int(n_keys)),
                            array<int, 2>({1, int(kv_stride)}));
        op_pv.run(tp_v, tv_v0, co0);
        if (PV_TILES > 1 && hd > PV_N) {
            const uint hd1 = min(PV_N, hd - PV_N);
            auto tv_v1 = tensor((device float*)(kv_row(v, paged, ring, t0, kv_stride) + kv_head_off + PV_N),
                                dextents<int32_t, 2>(int(hd1), int(n_keys)),
                                array<int, 2>({1, int(kv_stride)}));
            op_pv.run(tp_v, tv_v1, co1);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint rh = 0; rh < ROWS_PER_SG; ++rh) {
        const uint r = sg + rh * 4;
        if (lane == 0) tg_corr[r] = l_row[rh] > 0.0f ? (1.0f / l_row[rh]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint16_t i = 0; i < co0.get_capacity(); ++i) {
        if (co0.is_valid_element(i)) {
            auto idx = co0.get_multidimensional_index(i);
            const uint col = uint(idx[0]);
            const uint row = uint(idx[1]);
            const uint qi = q0 + row;
            if (qi < nq) {
                if (col < hd) {
                    out[(ulong)qi * q_stride + q_head_off + col] = co0[i] * tg_corr[row];
                }
                if (PV_TILES > 1 && PV_N + col < hd) {
                    out[(ulong)qi * q_stride + q_head_off + PV_N + col] = co1[i] * tg_corr[row];
                }
            }
        }
    }
}

// f16 KV cache path (`Step::CachedAttention` with `kv_f16 = true`): Q is staged once as half in
// threadgroup memory, while K and V are read directly as strided `device half` tensors.
template <uint HD, uint QB>
static inline void flash_attn_mm_f16(device const float* q, device const half* k, device const half* v,
                                     device float* out,
                                     uint nq, uint heads, uint kv_heads, uint head_dim,
                                     ulong q_stride, ulong kv_stride, float scale,
                                     bool has_cap, float soft_cap,
                                     uint row_lo_lane, uint row_hi_lane,
                                     threadgroup half* tq, threadgroup float* ts, threadgroup half* tp,
                                     threadgroup float* tg_corr,
                                     threadgroup uint* tg_row_lo, threadgroup uint* tg_row_hi,
                                     threadgroup uint* tg_bounds,
                                     uint2 tgp, uint sg, uint lane,
                                     uint paged, uint ring) {
    constexpr uint C = 32;
    constexpr uint ROWS_PER_SG = QB / 4;
    const uint hd = head_dim;
    const uint head = tgp.y;
    const uint kvh = head / (heads / kv_heads);
    const uint q0 = tgp.x * QB;
    if (q0 >= nq) return;
    const uint q_ext = min(QB, nq - q0);
    const uint tid = sg * 32 + lane;
    const ulong q_head_off = (ulong)head * hd;
    const ulong kv_head_off = (ulong)kvh * hd;

    if (sg == 0) {
        if (lane < QB) {
            tg_row_lo[lane] = row_lo_lane;
            tg_row_hi[lane] = row_hi_lane;
        }
        const uint min_lo = simd_min(row_lo_lane);
        const uint max_hi = simd_max(row_hi_lane);
        if (lane == 0) {
            tg_bounds[0] = min_lo;
            tg_bounds[1] = max_hi;
        }
    }
    for (uint idx = tid; idx < QB * HD; idx += 128) {
        const uint r = idx / HD;
        const uint d = idx % HD;
        const uint qi = q0 + r;
        tq[idx] = (qi < nq && d < hd)
            ? kv_store(q[(ulong)qi * q_stride + q_head_off + d], half(0))
            : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tg_lo = paged ? (tg_bounds[0] & ~(C - 1)) : tg_bounds[0];
    const uint tg_hi = tg_bounds[1];

    float m_row[ROWS_PER_SG];
    float l_row[ROWS_PER_SG];
    #pragma unroll
    for (uint rh = 0; rh < ROWS_PER_SG; ++rh) {
        m_row[rh] = -INFINITY;
        l_row[rh] = 0.0f;
    }

    auto tq_v = tensor(tq, dextents<int32_t, 2>(int(hd), int(q_ext)), array<int, 2>({1, int(HD)}));
    constexpr uint PV_N = (HD > 256) ? 256 : HD;
    constexpr uint PV_TILES = HD / PV_N;
    auto tp_proto = tensor(tp, dextents<int32_t, 2>(int(C), int(QB)), array<int, 2>({1, int(C)}));
    auto tv_proto = tensor((device half*)v, dextents<int32_t, 2>(int(PV_N), int(C)), array<int, 2>({1, int(kv_stride)}));

    constexpr auto desc_qk = matmul2d_descriptor(
        int(QB), int(C), static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc_qk, execution_simdgroups<4>> op_qk;
    constexpr auto desc_pv = matmul2d_descriptor(
        int(QB), int(PV_N), static_cast<int>(dynamic_extent), false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc_pv, execution_simdgroups<4>> op_pv;

    auto co0 = op_pv.template get_destination_cooperative_tensor<decltype(tp_proto), decltype(tv_proto), float>();
    auto co1 = op_pv.template get_destination_cooperative_tensor<decltype(tp_proto), decltype(tv_proto), float>();
    #pragma unroll
    for (uint16_t i = 0; i < co0.get_capacity(); ++i) {
        if (co0.is_valid_element(i)) co0[i] = 0.0f;
        if (PV_TILES > 1 && co1.is_valid_element(i)) co1[i] = 0.0f;
    }

    for (uint t0 = tg_lo; t0 < tg_hi; t0 += C) {
        const uint n_keys = min(C, tg_hi - t0);
        auto tk_v = tensor((device half*)(kv_row(k, paged, ring, t0, kv_stride) + kv_head_off),
                           dextents<int32_t, 2>(int(hd), int(n_keys)),
                           array<int, 2>({1, int(kv_stride)}));
        auto cs = op_qk.template get_destination_cooperative_tensor<decltype(tq_v), decltype(tk_v), float>();
        #pragma unroll
        for (uint16_t i = 0; i < cs.get_capacity(); ++i) {
            if (cs.is_valid_element(i)) cs[i] = 0.0f;
        }
        op_qk.run(tq_v, tk_v, cs);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma unroll
        for (uint16_t i = 0; i < cs.get_capacity(); ++i) {
            if (cs.is_valid_element(i)) {
                auto idx = cs.get_multidimensional_index(i);
                const uint col = uint(idx[0]);
                const uint row = uint(idx[1]);
                const uint ki = t0 + col;
                float s = -INFINITY;
                if (row < q_ext && col < n_keys && ki >= tg_row_lo[row] && ki < tg_row_hi[row]) {
                    s = cs[i] * scale;
                    if (has_cap) s = soft_cap * tanh(s / soft_cap);
                }
                if (row < QB && col < C) ts[row * C + col] = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint rh = 0; rh < ROWS_PER_SG; ++rh) {
            const uint r = sg + rh * 4;
            const float s = (r < q_ext && lane < n_keys) ? ts[r * C + lane] : -INFINITY;
            const float m_tile = simd_max(s);
            if (m_tile > -INFINITY) {
                const float m_old = m_row[rh];
                const float m_new = max(m_old, m_tile);
                const float corr = (m_old > -INFINITY) ? exp(m_old - m_new) : 0.0f;
                const float e = (s > -INFINITY) ? exp(s - m_new) : 0.0f;
                l_row[rh] = l_row[rh] * corr + simd_sum(e);
                m_row[rh] = m_new;
                tp[r * C + lane] = half(e);
                if (lane == 0) tg_corr[r] = corr;
            } else {
                tp[r * C + lane] = half(0.0h);
                if (lane == 0) tg_corr[r] = 1.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint16_t i = 0; i < co0.get_capacity(); ++i) {
            if (co0.is_valid_element(i)) {
                auto idx = co0.get_multidimensional_index(i);
                const uint row = uint(idx[1]);
                if (row < QB) {
                    co0[i] *= tg_corr[row];
                    if (PV_TILES > 1) co1[i] *= tg_corr[row];
                }
            }
        }
        auto tp_v = tensor(tp, dextents<int32_t, 2>(int(n_keys), int(q_ext)), array<int, 2>({1, int(C)}));
        const uint hd0 = min(PV_N, hd);
        auto tv_v0 = tensor((device half*)(kv_row(v, paged, ring, t0, kv_stride) + kv_head_off),
                            dextents<int32_t, 2>(int(hd0), int(n_keys)),
                            array<int, 2>({1, int(kv_stride)}));
        op_pv.run(tp_v, tv_v0, co0);
        if (PV_TILES > 1 && hd > PV_N) {
            const uint hd1 = min(PV_N, hd - PV_N);
            auto tv_v1 = tensor((device half*)(kv_row(v, paged, ring, t0, kv_stride) + kv_head_off + PV_N),
                                dextents<int32_t, 2>(int(hd1), int(n_keys)),
                                array<int, 2>({1, int(kv_stride)}));
            op_pv.run(tp_v, tv_v1, co1);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint rh = 0; rh < ROWS_PER_SG; ++rh) {
        const uint r = sg + rh * 4;
        if (lane == 0) tg_corr[r] = l_row[rh] > 0.0f ? (1.0f / l_row[rh]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint16_t i = 0; i < co0.get_capacity(); ++i) {
        if (co0.is_valid_element(i)) {
            auto idx = co0.get_multidimensional_index(i);
            const uint col = uint(idx[0]);
            const uint row = uint(idx[1]);
            const uint qi = q0 + row;
            if (qi < nq) {
                if (col < hd) {
                    out[(ulong)qi * q_stride + q_head_off + col] = co0[i] * tg_corr[row];
                }
                if (PV_TILES > 1 && PV_N + col < hd) {
                    out[(ulong)qi * q_stride + q_head_off + PV_N + col] = co1[i] * tg_corr[row];
                }
            }
        }
    }
}

#define FLASH_ATTN_MM_KERNEL(HD, QB)                                                             \
kernel void flash_attn_mm_##HD(device const float* q [[buffer(0)]],                              \
                               device const float* k [[buffer(1)]],                              \
                               device const float* v [[buffer(2)]],                              \
                               device const float* bias [[buffer(3)]],                           \
                               device float* out [[buffer(4)]],                                  \
                               constant AttnFlashParams& p [[buffer(5)]],                        \
                               uint2 tgp [[threadgroup_position_in_grid]],                       \
                               uint sg [[simdgroup_index_in_threadgroup]],                       \
                               uint lane [[thread_index_in_simdgroup]]) {                        \
    threadgroup float ts[QB * 32];                                                               \
    threadgroup float tg_corr[QB];                                                               \
    threadgroup uint tg_row_lo[QB];                                                              \
    threadgroup uint tg_row_hi[QB];                                                              \
    threadgroup uint tg_bounds[2];                                                               \
    uint lo = UINT_MAX, hi = 0;                                                                  \
    const uint qi = tgp.x * QB + lane;                                                           \
    if (lane < QB && qi < p.nq) {                                                                \
        attn_key_range(qi, p, lo, hi);                                                           \
        if (lo >= hi) { lo = UINT_MAX; hi = 0; }                                                 \
    }                                                                                            \
    const ulong q_stride = (ulong)p.heads * p.head_dim;                                          \
    const ulong kv_stride = (ulong)p.kv_heads * p.head_dim;                                      \
    flash_attn_mm_f32<HD, QB>(q, k, v, bias, out, p.nq, p.nk, p.heads, p.kv_heads,               \
                              p.head_dim, q_stride, kv_stride, p.scale,                          \
                              (p.flags & 4u) != 0, p.soft_cap, p.bias_mode, lo, hi,              \
                              ts, tg_corr, tg_row_lo, tg_row_hi, tg_bounds, tgp, sg, lane);      \
}

#define FLASH_ATTN_CACHED_MM_F32_KERNEL(NAME, HD, QB)                                            \
kernel void NAME(device const float* q [[buffer(0)]],                                            \
                 device const float* kc [[buffer(1)]],                                           \
                 device const float* vc [[buffer(2)]],                                           \
                 device float* out [[buffer(3)]],                                                \
                 constant AttnCachedParams& p [[buffer(4)]],                                     \
                 device const uint* vis [[buffer(5)]],                                           \
                 uint2 tgp [[threadgroup_position_in_grid]],                                     \
                 uint sg [[simdgroup_index_in_threadgroup]],                                     \
                 uint lane [[thread_index_in_simdgroup]]) {                                      \
    threadgroup float ts[QB * 32];                                                               \
    threadgroup float tg_corr[QB];                                                               \
    threadgroup uint tg_row_lo[QB];                                                              \
    threadgroup uint tg_row_hi[QB];                                                              \
    threadgroup uint tg_bounds[2];                                                               \
    uint lo = UINT_MAX, hi = 0;                                                                  \
    const uint qi = tgp.x * QB + lane;                                                           \
    if (lane < QB && qi < p.m) {                                                                 \
        const uint cur_pos = p.kv_base + qi;                                                     \
        hi = p.has_vis ? vis[qi] - (p.start_pos - p.kv_base) : (cur_pos + 1);                    \
        lo = (p.window > 0 && cur_pos + 1 > p.window) ? (cur_pos + 1 - p.window) : 0;            \
        if (lo >= hi) { lo = UINT_MAX; hi = 0; }                                                 \
    }                                                                                            \
    const uint kv_heads = p.n_heads / p.group;                                                   \
    const ulong q_stride = (ulong)p.n_heads * p.head_dim;                                        \
    flash_attn_mm_f32<HD, QB>(q, kc, vc, nullptr, out, p.m, 0, p.n_heads, kv_heads,              \
                              p.head_dim, q_stride, (ulong)p.stride, p.scale,                    \
                              p.softcap > 0.0f, p.softcap, 0, lo, hi,                            \
                              ts, tg_corr, tg_row_lo, tg_row_hi, tg_bounds, tgp, sg, lane,       \
                              p.paged, p.ring_pages);                                            \
}

#define FLASH_ATTN_CACHED_MM_F16_KERNEL(NAME, HD, QB)                                            \
kernel void NAME(device const float* q [[buffer(0)]],                                            \
                 device const half* kc [[buffer(1)]],                                            \
                 device const half* vc [[buffer(2)]],                                            \
                 device float* out [[buffer(3)]],                                                \
                 constant AttnCachedParams& p [[buffer(4)]],                                     \
                 device const uint* vis [[buffer(5)]],                                           \
                 uint2 tgp [[threadgroup_position_in_grid]],                                     \
                 uint sg [[simdgroup_index_in_threadgroup]],                                     \
                 uint lane [[thread_index_in_simdgroup]]) {                                      \
    threadgroup half tq[QB * HD];                                                                \
    threadgroup float ts[QB * 32];                                                               \
    threadgroup half tp[QB * 32];                                                                \
    threadgroup float tg_corr[QB];                                                               \
    threadgroup uint tg_row_lo[QB];                                                              \
    threadgroup uint tg_row_hi[QB];                                                              \
    threadgroup uint tg_bounds[2];                                                               \
    uint lo = UINT_MAX, hi = 0;                                                                  \
    const uint qi = tgp.x * QB + lane;                                                           \
    if (lane < QB && qi < p.m) {                                                                 \
        const uint cur_pos = p.kv_base + qi;                                                     \
        hi = p.has_vis ? vis[qi] - (p.start_pos - p.kv_base) : (cur_pos + 1);                    \
        lo = (p.window > 0 && cur_pos + 1 > p.window) ? (cur_pos + 1 - p.window) : 0;            \
        if (lo >= hi) { lo = UINT_MAX; hi = 0; }                                                 \
    }                                                                                            \
    const uint kv_heads = p.n_heads / p.group;                                                   \
    const ulong q_stride = (ulong)p.n_heads * p.head_dim;                                        \
    flash_attn_mm_f16<HD, QB>(q, kc, vc, out, p.m, p.n_heads, kv_heads, p.head_dim,              \
                              q_stride, (ulong)p.stride, p.scale,                                \
                              p.softcap > 0.0f, p.softcap, lo, hi,                               \
                              tq, ts, tp, tg_corr, tg_row_lo, tg_row_hi, tg_bounds,              \
                              tgp, sg, lane, p.paged, p.ring_pages);                             \
}

FLASH_ATTN_MM_KERNEL(32, 32)
FLASH_ATTN_MM_KERNEL(64, 32)
FLASH_ATTN_MM_KERNEL(128, 32)
FLASH_ATTN_MM_KERNEL(256, 32)
FLASH_ATTN_MM_KERNEL(512, 16)

FLASH_ATTN_CACHED_MM_F32_KERNEL(flash_attn_cached_mm_f32_32, 32, 32)
FLASH_ATTN_CACHED_MM_F32_KERNEL(flash_attn_cached_mm_f32_64, 64, 32)
FLASH_ATTN_CACHED_MM_F32_KERNEL(flash_attn_cached_mm_f32_128, 128, 32)
FLASH_ATTN_CACHED_MM_F32_KERNEL(flash_attn_cached_mm_f32_256, 256, 32)
FLASH_ATTN_CACHED_MM_F32_KERNEL(flash_attn_cached_mm_f32_512, 512, 16)
FLASH_ATTN_CACHED_MM_F16_KERNEL(flash_attn_cached_mm_f16_32, 32, 32)
FLASH_ATTN_CACHED_MM_F16_KERNEL(flash_attn_cached_mm_f16_64, 64, 32)
FLASH_ATTN_CACHED_MM_F16_KERNEL(flash_attn_cached_mm_f16_128, 128, 32)
FLASH_ATTN_CACHED_MM_F16_KERNEL(flash_attn_cached_mm_f16_256, 256, 16)
FLASH_ATTN_CACHED_MM_F16_KERNEL(flash_attn_cached_mm_f16_512, 512, 8)
#endif

// ---- RelPosBias ---------------------------------------------------------------------------------
// out[h, i, j] = table[bucket(k_pos[j] - q_pos[i]), h]; HF `_relative_position_bucket`, the same f32
// arithmetic as `cpu::rel_pos_bucket` (`scale` = ln(max_distance / exact) rounded from f64).
struct RelPosParams {
    uint n;
    uint m;
    uint heads;
    uint buckets;
    uint bidir;
    float scale;
};

static inline uint rel_pos_bucket(long rel, constant RelPosParams& p) {
    uint nb = p.buckets;
    uint ret = 0;
    uint n;
    if (p.bidir != 0) {
        nb /= 2;
        if (rel > 0) ret += nb;
        n = uint(rel < 0 ? -rel : rel);
    } else {
        n = rel < 0 ? uint(-rel) : 0u;
    }
    const uint exact = nb / 2;
    if (n < exact) return ret + n;
    const uint large = exact + uint(log(float(n) / float(exact)) / p.scale * float(nb - exact));
    return ret + min(large, nb - 1);
}

// One thread per (query, key); writes that position for every head.
kernel void rel_pos_bias_f32(device const float* table [[buffer(0)]],
                             device const uint* q_pos [[buffer(1)]],
                             device const uint* k_pos [[buffer(2)]],
                             device float* out [[buffer(3)]],
                             constant RelPosParams& p [[buffer(4)]],
                             uint i [[thread_position_in_grid]]) {
    const ulong nm = (ulong)p.n * p.m;
    if (i >= nm) return;
    const uint qi = i / p.m;
    const uint ki = i - qi * p.m;
    const uint b = rel_pos_bucket(long(k_pos[ki]) - long(q_pos[qi]), p);
    for (uint h = 0; h < p.heads; ++h) out[(ulong)h * nm + i] = table[b * p.heads + h];
}

// ---- StateReorder -------------------------------------------------------------------------------
// out row b = state row parent[b], or zeros when parent[b] >= batch (as the CPU executor).
struct ReorderParams {
    uint batch;
    uint row;
};

kernel void state_reorder_f32(device const float* state [[buffer(0)]],
                              device const uint* parent [[buffer(1)]],
                              device float* out [[buffer(2)]],
                              constant ReorderParams& p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if ((ulong)i >= (ulong)p.batch * p.row) return;
    const uint b = i / p.row;
    const uint c = i - b * p.row;
    const uint src = parent[b];
    out[i] = src < p.batch ? state[(ulong)src * p.row + c] : 0.0f;
}

// ---- MRope --------------------------------------------------------------------------------------
// One thread per element of `[n_seq, n_heads * head_dim]`; positions are `[n_seq, 3]` (t, h, w).
// Pair `j` takes its position from the t/h/w section (contiguous `s0, s1, s2` pairs, or the
// interleaved `j % 3` pattern), as the CPU executor. Adjacent pairs `(2j, 2j+1)` or half-split
// `(j, j + rotary/2)`.
struct MRopeParams {
    uint n_seq;
    uint n_heads;
    uint head_dim;
    uint rotary_dim;
    uint adjacent;
    uint interleaved;
    uint s0;
    uint s1;
    uint s2;
    float mscale;
};

kernel void rope_mrope_f32(device const float* in [[buffer(0)]],
                           device const uint* pos [[buffer(1)]],
                           device const float* inv_freq [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant MRopeParams& p [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    const ulong total = (ulong)p.n_seq * p.n_heads * p.head_dim;
    if ((ulong)gid >= total) return;
    const uint d = gid % p.head_dim;
    const ulong rem = gid / p.head_dim;
    const uint seq = uint(rem / p.n_heads);
    const ulong base = gid - d;
    if (d >= p.rotary_dim) {
        out[gid] = in[gid];
        return;
    }
    const uint half_dim = p.rotary_dim / 2;
    uint j, partner;
    bool first;
    if (p.adjacent != 0) {
        j = d / 2;
        first = (d % 2) == 0;
        partner = d ^ 1;
    } else {
        first = d < half_dim;
        j = first ? d : d - half_dim;
        partner = first ? d + half_dim : d - half_dim;
    }
    uint axis;
    if (p.interleaved != 0) {
        axis = (j % 3 == 1 && j < 3 * p.s1) ? 1u : ((j % 3 == 2 && j < 3 * p.s2) ? 2u : 0u);
    } else {
        axis = j < p.s0 ? 0u : (j < p.s0 + p.s1 ? 1u : 2u);
    }
    const float theta = float(pos[seq * 3 + axis]) * inv_freq[j];
    const float c = cos(theta) * p.mscale;
    const float s = sin(theta) * p.mscale;
    const float a = in[gid];
    const float b = in[base + partner];
    out[gid] = first ? a * c - b * s : b * s + a * c;
}
