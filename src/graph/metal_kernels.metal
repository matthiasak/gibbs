// Graph-IR kernels for the Metal executor (`src/graph/metal.rs`), all f32. Compiled at runtime by
// `newLibraryWithSource`. `metal.rs` first compiles with `GIBBS_MPP` defined and MSL 4.0, which adds
// the MetalPerformancePrimitives GEMM; if that fails, without it at the default language version.
// Each kernel mirrors a `crate::model::nn` op; `metal.rs` tests compare them against it.
#include <metal_stdlib>
#if defined(GIBBS_MPP)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

// Attention: head_dim <= 32 * ATTN_MAX_PER_LANE; simdgroups per threadgroup <= ATTN_MAX_SG.
constant uint ATTN_MAX_PER_LANE = 16;
constant uint ATTN_MAX_SG = 8;

struct EmbedParams { uint n; uint d; uint rows; };
struct NormParams { uint d; float eps; };
struct AttnParams { uint nq; uint nk; uint heads; uint head_dim; uint causal; float scale; };
struct PoolParams { uint n; uint d; };

// GEMM-COMMON-BEGIN
// GEMV-COMMON-BEGIN (also prepended to generated GEMV kernels by `codegen::gemv_packed`)
// Tanh-approximated GELU (`gelu_pytorch_tanh`), the activation of GeGlu.
static inline float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Exact GELU with the Numerical Recipes erfc fit used by `nn::gelu` (|erf error| < 1.2e-7), in f32.
static inline float gelu_erf(float x) {
    float z = fabs(x) * M_SQRT1_2_F;
    float t = 1.0f / (1.0f + 0.5f * z);
    float poly = -z * z - 1.26551223f
        + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f
        + t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    float erfc_v = t * exp(poly);
    float erf_v = x >= 0.0f ? 1.0f - erfc_v : erfc_v - 1.0f;
    return 0.5f * x * (1.0f + erf_v);
}

struct PackedGemvParams {
    uint rows;
    uint cols;
    uint batch_size;
    uint has_bias;
    uint epi; // 0 = None, 1 = Gelu, 2 = Residual, 3 = Silu, 4 = SwiGlu, 5 = GeGlu
};

static inline float apply_gemv_epi(float acc, ulong out_idx, constant PackedGemvParams& p, device const float* res) {
    if (p.epi == 1) {
        return gelu_erf(acc);
    } else if (p.epi == 2) {
        return acc + res[out_idx];
    } else if (p.epi == 3) {
        return acc / (1.0f + exp(-acc));
    } else if (p.epi == 4) {
        float silu = acc / (1.0f + exp(-acc));
        return silu * res[out_idx];
    } else if (p.epi == 5) {
        return gelu_tanh(acc) * res[out_idx];
    } else if (p.epi == 6) {
        float gate = res[out_idx];
        return acc * (gate / (1.0f + exp(-gate)));
    } else if (p.epi == 7) {
        return acc * gelu_tanh(res[out_idx]);
    }
    return acc;
}
// GEMV-COMMON-END

struct GemmParams { uint m; uint n; uint k; uint has_bias; };

// GEMM epilogues, applied to x·Wᵀ (+ bias) before the store.
enum : int { EPI_NONE = 0, EPI_GELU = 1, EPI_RESIDUAL = 2 };

template <int EPI>
static inline float epilogue(float acc, uint row, uint col, constant GemmParams& p,
                             device const float* bias, device const float* res) {
    float v = p.has_bias ? acc + bias[col] : acc;
    if (EPI == EPI_GELU) v = gelu_erf(v);
    if (EPI == EPI_RESIDUAL) v += res[(ulong)row * p.n + col];
    return v;
}
// GEMM-COMMON-END

// Sum across the whole threadgroup; every thread gets the total. Safe to call repeatedly.
static inline float block_sum(float v, threadgroup float* scratch, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < nsg; ++i) total += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

// out[m, n] = epilogue(x[m, k] · w[n, k]ᵀ): a 32×32 output tile per threadgroup of 4 simdgroups,
// each owning a 16×16 quadrant as 2×2 simdgroup_float8x8 accumulators. K advances 8 at a time
// through threadgroup memory, zero-filled past the matrix edges. Runs on every Apple GPU.
template <int EPI>
kernel void gemm_sg(device const float* x [[buffer(0)]],
                    device const float* w [[buffer(1)]],
                    device const float* bias [[buffer(2)]],
                    device const float* res [[buffer(3)]],
                    device float* out [[buffer(4)]],
                    constant GemmParams& p [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[32 * 8];  // [row][k]
    threadgroup float ws[8 * 32];  // [k][col]
    threadgroup float cs[32 * 32];
    const uint m0 = tg.y * 32, n0 = tg.x * 32;
    const uint sm = (sg / 2) * 16, sn = (sg % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = 0; k0 < p.k; k0 += 8) {
        for (uint e = tid; e < 256; e += 128) {
            // Consecutive threads read consecutive k of one row of x and of w.
            uint r = e / 8, kk = e % 8, gk = k0 + kk;
            xs[e] = (m0 + r < p.m && gk < p.k) ? x[(ulong)(m0 + r) * p.k + gk] : 0.0f;
            ws[kk * 32 + r] = (n0 + r < p.n && gk < p.k) ? w[(ulong)(n0 + r) * p.k + gk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a[2], b[2];
        for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], xs + (sm + 8 * i) * 8, 8);
        for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], ws + sn + 8 * j, 32);
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], cs + (sm + 8 * i) * 32 + sn + 8 * j, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 1024; e += 128) {
        uint r = m0 + e / 32, c = n0 + e % 32;
        if (r < p.m && c < p.n) out[(ulong)r * p.n + c] = epilogue<EPI>(cs[e], r, c, p, bias, res);
    }
}

#define GEMM_INSTANCE(name, kern, epi) \
    template [[host_name(name)]] [[kernel]] decltype(kern<epi>) kern<epi>;
GEMM_INSTANCE("gemm_sg_none", gemm_sg, EPI_NONE)
GEMM_INSTANCE("gemm_sg_gelu", gemm_sg, EPI_GELU)
GEMM_INSTANCE("gemm_sg_residual", gemm_sg, EPI_RESIDUAL)

// Same contract for few rows (decode-like m): one simdgroup per output column, i.e. per row of w,
// which is read once per 8 rows of x; 8 columns per threadgroup. Lanes take float4 chunks of k
// (lane 4i.., then +128), any k % 4 tail goes to lanes 0..2, and simd_sum reduces.
template <int EPI>
kernel void gemm_rows(device const float* x [[buffer(0)]],
                      device const float* w [[buffer(1)]],
                      device const float* bias [[buffer(2)]],
                      device const float* res [[buffer(3)]],
                      device float* out [[buffer(4)]],
                      constant GemmParams& p [[buffer(5)]],
                      uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    const uint col = tg * 8 + sg;
    if (col >= p.n) return;  // uniform per simdgroup
    device const float* wr = w + (ulong)col * p.k;
    const uint k4 = p.k & ~3u;
    for (uint r0 = 0; r0 < p.m; r0 += 8) {
        const uint rows = min(8u, p.m - r0);
        device const float* xr = x + (ulong)r0 * p.k;
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint i = lane * 4; i < k4; i += 128) {
            float4 wv = float4(*(device const packed_float4*)(wr + i));
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                if (j < rows) acc[j] += dot(wv, float4(*(device const packed_float4*)(xr + (ulong)j * p.k + i)));
            }
        }
        for (uint i = k4 + lane; i < p.k; i += 32) {
            float wv = wr[i];
            for (uint j = 0; j < rows; ++j) acc[j] += wv * xr[(ulong)j * p.k + i];
        }
        #pragma unroll
        for (uint j = 0; j < 8; ++j) {
            float s = simd_sum(acc[j]);
            if (lane == j && j < rows) out[(ulong)(r0 + j) * p.n + col] = epilogue<EPI>(s, r0 + j, col, p, bias, res);
        }
    }
}

GEMM_INSTANCE("gemm_rows_none", gemm_rows, EPI_NONE)
GEMM_INSTANCE("gemm_rows_gelu", gemm_rows, EPI_GELU)
GEMM_INSTANCE("gemm_rows_residual", gemm_rows, EPI_RESIDUAL)

#if defined(GIBBS_MPP)
using namespace mpp::tensor_ops;

// Same contract as gemm_sg on the GPU's matrix units: MetalPerformancePrimitives `matmul2d`,
// a 64×32 output tile per threadgroup of 4 simdgroups, epilogue applied in registers on the
// cooperative destination tensor before one store (edges are clipped by the tensor extents).
template <int EPI>
kernel void gemm_mpp(device float* x [[buffer(0)]],
                     device float* w [[buffer(1)]],
                     device const float* bias [[buffer(2)]],
                     device const float* res [[buffer(3)]],
                     device float* out [[buffer(4)]],
                     constant GemmParams& p [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]]) {
    // extent(0) is the contiguous dimension: x is [m][k], w is [n][k] (so right-transposed), out [m][n].
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tx(x, dextents<int32_t, 2>(p.k, p.m));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tw(w, dextents<int32_t, 2>(p.k, p.n));
    tensor<device float, dextents<int32_t, 2>, tensor_inline> to(out, dextents<int32_t, 2>(p.n, p.m));
    constexpr auto desc = matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mx = tx.slice(0, tg.y * 64);
    auto mw = tw.slice(0, tg.x * 32);
    auto mo = to.slice(tg.x * 32, tg.y * 64);
    auto c = op.get_destination_cooperative_tensor<decltype(mx), decltype(mw), float>();
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (c.is_valid_element(i)) c[i] = 0.0f;
    }
    op.run(mx, mw, c);
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {
        if (c.is_valid_element(i)) {
            auto idx = c.get_multidimensional_index(i);
            uint col = tg.x * 32 + uint(idx[0]), row = tg.y * 64 + uint(idx[1]);
            if (row < p.m && col < p.n) c[i] = epilogue<EPI>(c[i], row, col, p, bias, res);
        }
    }
    c.store(mo);
}

GEMM_INSTANCE("gemm_mpp_none", gemm_mpp, EPI_NONE)
GEMM_INSTANCE("gemm_mpp_gelu", gemm_mpp, EPI_GELU)
GEMM_INSTANCE("gemm_mpp_residual", gemm_mpp, EPI_RESIDUAL)
#endif

// out[r, :] = table[ids[r], :]; host ids are range-checked before dispatch, device-fed ids clamp.
kernel void embedding_f32(device const float* table [[buffer(0)]],
                          device const uint* ids [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant EmbedParams& p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.n * p.d) return;
    out[i] = table[(ulong)min(ids[i / p.d], p.rows - 1) * p.d + i % p.d];
}

// LayerNorm of one row per threadgroup: mean, then biased variance, like `nn::layer_norm`.
kernel void layer_norm_f32(device const float* x [[buffer(0)]],
                           device const float* g [[buffer(1)]],
                           device const float* b [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           constant NormParams& p [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tpg [[threads_per_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float* xr = x + (ulong)row * p.d;
    device float* o = out + (ulong)row * p.d;
    float s = 0.0f;
    for (uint i = tid; i < p.d; i += tpg) s += xr[i];
    float mean = block_sum(s, scratch, sg, lane, nsg) / float(p.d);
    float sq = 0.0f;
    for (uint i = tid; i < p.d; i += tpg) {
        float t = xr[i] - mean;
        sq += t * t;
    }
    float inv = 1.0f / sqrt(block_sum(sq, scratch, sg, lane, nsg) / float(p.d) + p.eps);
    for (uint i = tid; i < p.d; i += tpg) o[i] = (xr[i] - mean) * inv * g[i] + b[i];
}

// Mean of the rows whose mask entry is non-zero (count clamped to 1), one thread per column.
kernel void mean_pool_f32(device const float* x [[buffer(0)]],
                          device const uint* mask [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant PoolParams& p [[buffer(3)]],
                          uint c [[thread_position_in_grid]]) {
    if (c >= p.d) return;
    float s = 0.0f;
    uint count = 0;
    for (uint r = 0; r < p.n; ++r) {
        if (mask[r] != 0) {
            s += x[(ulong)r * p.d + c];
            ++count;
        }
    }
    out[c] = s / float(max(count, 1u));
}

// x / max(‖x‖₂, 1e-12) in one threadgroup.
kernel void l2_normalize_f32(device const float* x [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             constant uint& n [[buffer(2)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tpg [[threads_per_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float scratch[32];
    float s = 0.0f;
    for (uint i = tid; i < n; i += tpg) s += x[i] * x[i];
    float norm = max(sqrt(block_sum(s, scratch, sg, lane, nsg)), 1e-12f);
    for (uint i = tid; i < n; i += tpg) out[i] = x[i] / norm;
}

// Host profile: GPU cost of one dispatch boundary (an empty kernel back to back).
kernel void noop(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0xffffffffu) x[0] = 0.0f;
}

// Host profile: Streaming read throughput benchmark sweeping a memory buffer.
kernel void memory_sweep(device const ulong2* in [[buffer(0)]],
                         device atomic_uint* out [[buffer(1)]],
                         constant uint& count [[buffer(2)]],
                         uint tid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
    ulong2 s = 0;
    uint idx = tid * 4;
    if (idx + 3 < count) {
        s += in[idx];
        s += in[idx + 1];
        s += in[idx + 2];
        s += in[idx + 3];
    }
    uint acc = uint(s.x ^ s.y);
    acc = simd_sum(acc);
    if (lane == 0) {
        atomic_fetch_add_explicit(out, acc, memory_order_relaxed);
    }
}

// 2x ulong4 variant
kernel void memory_sweep_ulong4(device const ulong4* in [[buffer(0)]],
                                device atomic_uint* out [[buffer(1)]],
                                constant uint& count [[buffer(2)]],
                                uint tid [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    ulong4 s = 0;
    uint idx = tid * 2;
    if (idx + 1 < count) {
        s += in[idx];
        s += in[idx + 1];
    }
    uint acc = uint(s.x ^ s.y ^ s.z ^ s.w);
    acc = simd_sum(acc);
    if (lane == 0) {
        atomic_fetch_add_explicit(out, acc, memory_order_relaxed);
    }
}

// Grid-stride 16 B/thread variant with 4 iterations per thread
kernel void memory_sweep_grid_stride(device const ulong2* in [[buffer(0)]],
                                     device atomic_uint* out [[buffer(1)]],
                                     constant uint& count [[buffer(2)]],
                                     uint tid [[thread_position_in_grid]],
                                     uint grid_size [[threads_per_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    ulong2 s = 0;
    for (uint idx = tid; idx < count; idx += grid_size) {
        s += in[idx];
    }
    uint acc = uint(s.x ^ s.y);
    acc = simd_sum(acc);
    if (lane == 0) {
        atomic_fetch_add_explicit(out, acc, memory_order_relaxed);
    }
}

struct RmsNormParams { uint rows; uint d; float eps; };
kernel void rmsnorm_f32(device const float* x [[buffer(0)]],
                        device const float* g [[buffer(1)]],
                        device float* out [[buffer(2)]],
                        constant RmsNormParams& p [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint nsg [[simdgroups_per_threadgroup]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    uint row = tg;
    if (row >= p.rows) return;
    device const float* xr = x + (ulong)row * p.d;
    device float* outr = out + (ulong)row * p.d;
    float sq = 0.0f;
    for (uint i = tid; i < p.d; i += tpg) {
        float v = xr[i];
        sq += v * v;
    }
    float scale = rsqrt(block_sum(sq, scratch, sg, lane, nsg) / float(p.d) + p.eps);
    for (uint i = tid; i < p.d; i += tpg) {
        outr[i] = xr[i] * scale * g[i];
    }
}
inline void rmsnorm_rope_row(device const float* xr,
                             device const float* g,
                             device float* outr,
                             float scale,
                             uint d,
                             uint row,
                             uint n_heads,
                             uint rotary_dim,
                             uint pair_stride,
                             float mscale,
                             device const uint* pos,
                             device const float* inv_freq,
                             uint tid,
                             uint tpg) {
    uint seq = row / max(n_heads, 1u);
    uint p_val = pos[seq];
    uint half_dim = rotary_dim / 2;
    for (uint i = tid; i < d; i += tpg) {
        if (i < rotary_dim) {
            if (pair_stride == 2) {
                uint pair_idx = i / 2;
                float freq = inv_freq[pair_idx];
                float a = xr[pair_idx * 2] * scale * g[pair_idx * 2];
                float b = xr[pair_idx * 2 + 1] * scale * g[pair_idx * 2 + 1];
                if (freq == 0.0f) {
                    outr[i] = ((i % 2 == 0) ? a : b) * mscale;
                } else {
                    float theta = float(p_val) * freq;
                    float cos_th = cos(theta) * mscale;
                    float sin_th = sin(theta) * mscale;
                    outr[i] = (i % 2 == 0) ? (a * cos_th - b * sin_th) : (b * cos_th + a * sin_th);
                }
            } else if (i < half_dim) {
                float freq = inv_freq[i];
                float a = xr[i] * scale * g[i];
                float b = xr[i + half_dim] * scale * g[i + half_dim];
                if (freq == 0.0f) {
                    outr[i] = a * mscale;
                    outr[i + half_dim] = b * mscale;
                } else {
                    float theta = float(p_val) * freq;
                    float cos_th = cos(theta) * mscale;
                    float sin_th = sin(theta) * mscale;
                    outr[i] = a * cos_th - b * sin_th;
                    outr[i + half_dim] = b * cos_th + a * sin_th;
                }
            }
        } else {
            outr[i] = xr[i] * scale * g[i];
        }
    }
}

// Horizontal batch of up to 4 independent RmsNorm segments (own x, gamma, out, rows, d, eps each);
// threadgroup `tg` handles one row of the segment its row index falls in.
struct RmsNormBatchParams {
    uint rows[4];
    uint d[4];
    float eps[4];
    uint nseg;
    uint rope_mask;
    uint n_seq;
    uint rotary_dim;
    uint pair_stride;
    float mscale;
};
kernel void rmsnorm_batch_f32(device const float* x0 [[buffer(0)]], device const float* g0 [[buffer(1)]], device float* o0 [[buffer(2)]],
                              device const float* x1 [[buffer(3)]], device const float* g1 [[buffer(4)]], device float* o1 [[buffer(5)]],
                              device const float* x2 [[buffer(6)]], device const float* g2 [[buffer(7)]], device float* o2 [[buffer(8)]],
                              device const float* x3 [[buffer(9)]], device const float* g3 [[buffer(10)]], device float* o3 [[buffer(11)]],
                              constant RmsNormBatchParams& p [[buffer(12)]],
                              device const uint* pos [[buffer(13)]],
                              device const float* inv_freq [[buffer(14)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint nsg [[simdgroups_per_threadgroup]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    device const float* xs[4] = {x0, x1, x2, x3};
    device const float* gs[4] = {g0, g1, g2, g3};
    device float* os[4] = {o0, o1, o2, o3};
    uint row = tg;
    uint s = 0;
    while (s < p.nseg && row >= p.rows[s]) { row -= p.rows[s]; s++; }
    if (s >= p.nseg) return;
    uint d = p.d[s];
    device const float* xr = xs[s] + (ulong)row * d;
    device float* outr = os[s] + (ulong)row * d;
    float sq = 0.0f;
    for (uint i = tid; i < d; i += tpg) {
        float v = xr[i];
        sq += v * v;
    }
    float scale = rsqrt(block_sum(sq, scratch, sg, lane, nsg) / float(d) + p.eps[s]);
    if (p.rope_mask & (1u << s)) {
        rmsnorm_rope_row(xr, gs[s], outr, scale, d, row, p.rows[s] / max(p.n_seq, 1u),
                         p.rotary_dim, p.pair_stride, p.mscale, pos, inv_freq, tid, tpg);
    } else {
        for (uint i = tid; i < d; i += tpg) {
            outr[i] = xr[i] * scale * gs[s][i];
        }
    }
}
kernel void rmsnorm_residual_f32(device const float* x [[buffer(0)]],
                                 device const float* g [[buffer(1)]],
                                 device const float* res [[buffer(2)]],
                                 device float* out [[buffer(3)]],
                                 constant RmsNormParams& p [[buffer(4)]],
                                 uint tg [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint nsg [[simdgroups_per_threadgroup]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    uint row = tg;
    if (row >= p.rows) return;
    device const float* xr = x + (ulong)row * p.d;
    device const float* resr = res + (ulong)row * p.d;
    device float* outr = out + (ulong)row * p.d;
    float sq = 0.0f;
    for (uint i = tid; i < p.d; i += tpg) {
        float v = xr[i];
        sq += v * v;
    }
    float scale = rsqrt(block_sum(sq, scratch, sg, lane, nsg) / float(p.d) + p.eps);
    for (uint i = tid; i < p.d; i += tpg) {
        outr[i] = resr[i] + xr[i] * scale * g[i];
    }
}
struct RmsNormFusedParams {
    uint rows;
    uint d;
    float eps;
    float eps2;
    uint flags;
    uint n_seq;
    uint rotary_dim;
    uint pair_stride;
    float mscale;
};
kernel void rmsnorm_fused_f32(device const float* x [[buffer(0)]],
                              device const float* g [[buffer(1)]],
                              device const float* res [[buffer(2)]],
                              device const float* scale_buf [[buffer(3)]],
                              device float* out [[buffer(4)]],
                              device const float* g2 [[buffer(5)]],
                              device float* out2 [[buffer(6)]],
                              constant RmsNormFusedParams& p [[buffer(7)]],
                              device const uint* pos [[buffer(8)]],
                              device const float* inv_freq [[buffer(9)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint nsg [[simdgroups_per_threadgroup]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint tpg [[threads_per_threadgroup]]) {
    threadgroup float scratch[32];
    uint row = tg;
    if (row >= p.rows) return;
    device const float* xr = x + (ulong)row * p.d;
    device const float* resr = res + (ulong)row * p.d;
    device float* outr = out + (ulong)row * p.d;
    float sq = 0.0f;
    for (uint i = tid; i < p.d; i += tpg) {
        float v = xr[i];
        sq += v * v;
    }
    float inv = rsqrt(block_sum(sq, scratch, sg, lane, nsg) / float(p.d) + p.eps);
    float mul = (p.flags & 2u) ? scale_buf[0] : 1.0f;
    bool has_res = (p.flags & 1u) != 0;
    if (p.flags & 8u) {
        rmsnorm_rope_row(xr, g, outr, inv, p.d, row, p.rows / max(p.n_seq, 1u),
                         p.rotary_dim, p.pair_stride, p.mscale, pos, inv_freq, tid, tpg);
    } else if (p.flags & 4u) {
        device float* out2r = out2 + (ulong)row * p.d;
        float sq2 = 0.0f;
        for (uint i = tid; i < p.d; i += tpg) {
            float v = xr[i] * inv * g[i];
            if (has_res) v += resr[i];
            v *= mul;
            outr[i] = v;
            sq2 += v * v;
        }
        float inv2 = rsqrt(block_sum(sq2, scratch + 8, sg, lane, nsg) / float(p.d) + p.eps2);
        for (uint i = tid; i < p.d; i += tpg) {
            out2r[i] = outr[i] * inv2 * g2[i];
        }
    } else {
        for (uint i = tid; i < p.d; i += tpg) {
            float v = xr[i] * inv * g[i];
            if (has_res) v += resr[i];
            outr[i] = v * mul;
        }
    }
}


struct GraphRopeParams { uint n_heads; uint head_dim; uint pairs; uint pair_stride; uint pair_gap; };
kernel void graph_rope_f32(device const float* x [[buffer(0)]],
                           device float* out [[buffer(1)]],
                           constant float* cs [[buffer(2)]],
                           constant GraphRopeParams& p [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    uint total = p.n_heads * p.head_dim;
    if (gid >= total) return;
    uint head = gid / p.head_dim;
    uint d = gid % p.head_dim;
    uint pairs = p.pairs;
    uint i0 = head * p.head_dim + d;
    if (d < pairs * 2) {
        uint pair_idx = d / 2;
        float c = cs[pair_idx];
        float s = cs[pairs + pair_idx];
        if (d % 2 == 0) {
            float a = x[head * p.head_dim + pair_idx * 2];
            float b = x[head * p.head_dim + pair_idx * 2 + 1];
            out[i0] = a * c - b * s;
        } else {
            float a = x[head * p.head_dim + pair_idx * 2];
            float b = x[head * p.head_dim + pair_idx * 2 + 1];
            out[i0] = b * c + a * s;
        }
    } else {
        out[i0] = x[i0];
    }
}
kernel void swiglu_f32(device const float* gate [[buffer(0)]],
                       device const float* up [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float g = gate[i];
    out[i] = (g / (1.0f + exp(-g))) * up[i];
}

kernel void geglu_f32(device const float* gate [[buffer(0)]],
                      device const float* up [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    out[i] = gelu_tanh(gate[i]) * up[i];
}

kernel void split_glu_f32(device const float* gu [[buffer(0)]],
                          device float* out [[buffer(1)]],
                          constant uint4& p [[buffer(2)]],
                          uint idx [[thread_position_in_grid]]) {
    if (idx >= p.x * p.y) return;
    uint r = idx / p.y;
    uint c = idx % p.y;
    float g = gu[(ulong)r * (2u * p.y) + c];
    float u = gu[(ulong)r * (2u * p.y) + p.y + c];
    float act = p.z == 0u ? gelu_erf(g) : (p.z == 1u ? gelu_tanh(g) : (g / (1.0f + exp(-g))));
    out[idx] = act * u;
}

kernel void moe_weighted_sum_f32(device const float* down [[buffer(0)]],
                                 device const float* weights [[buffer(1)]],
                                 device float* out [[buffer(2)]],
                                 constant uint3& p [[buffer(3)]],
                                 uint idx [[thread_position_in_grid]]) {
    if (idx >= p.x * p.z) return;
    uint r = idx / p.z;
    uint c = idx % p.z;
    float sum = 0.0f;
    ulong base = (ulong)r * p.y;
    for (uint s = 0; s < p.y; ++s) {
        sum += down[(base + s) * p.z + c] * weights[base + s];
    }
    out[idx] = sum;
}

struct block_q4_0 {
    half d;
    uint8_t qs[16];
};


kernel void gemv_q4_0(device const block_q4_0* w [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant PackedGemvParams& p [[buffer(3)]],
                      device const float* bias [[buffer(4)]],
                      device const float* res [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 32;
    device const block_q4_0* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q4_0* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint b = lane; b < n_blocks; b += 32) {
        block_q4_0 blk0 = r0_blocks[b];
        block_q4_0 blk1 = has_row1 ? r1_blocks[b] : blk0;
        float d0 = float(blk0.d);
        float d1 = float(blk1.d);
        device const float* xb = xb_base + b * 32;

        float b_acc0 = 0.0f;
        float b_acc1 = 0.0f;

        #pragma unroll
        for (uint j = 0; j < 16; j += 4) {
            uint32_t q0 = *reinterpret_cast<thread const uint32_t*>(&blk0.qs[j]);
            uint32_t q1 = *reinterpret_cast<thread const uint32_t*>(&blk1.qs[j]);

            float4 v0_0 = float4(int4((q0 & 0x0F), ((q0 >> 8) & 0x0F), ((q0 >> 16) & 0x0F), ((q0 >> 24) & 0x0F)) - 8);
            float4 v1_0 = float4(int4(((q0 >> 4) & 0x0F), ((q0 >> 12) & 0x0F), ((q0 >> 20) & 0x0F), ((q0 >> 28) & 0x0F)) - 8);

            float4 v0_1 = float4(int4((q1 & 0x0F), ((q1 >> 8) & 0x0F), ((q1 >> 16) & 0x0F), ((q1 >> 24) & 0x0F)) - 8);
            float4 v1_1 = float4(int4(((q1 >> 4) & 0x0F), ((q1 >> 12) & 0x0F), ((q1 >> 20) & 0x0F), ((q1 >> 28) & 0x0F)) - 8);

            float4 x0 = *reinterpret_cast<device const float4*>(xb + j);
            float4 x1 = *reinterpret_cast<device const float4*>(xb + j + 16);

            b_acc0 += dot(v0_0, x0) + dot(v1_0, x1);
            b_acc1 += dot(v0_1, x0) + dot(v1_1, x1);
        }
        acc0 += d0 * b_acc0;
        if (has_row1) {
            acc1 += d1 * b_acc1;
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        if (has_row1) acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_q4_0_half(device const block_q4_0* w [[buffer(0)]],
                           device const float* x [[buffer(1)]],
                           device float* y [[buffer(2)]],
                           constant PackedGemvParams& p [[buffer(3)]],
                           device const float* bias [[buffer(4)]],
                           device const float* res [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 32;
    uint n_sub = n_blocks * 2;
    device const block_q4_0* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q4_0* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint s = lane; s < n_sub; s += 32) {
        uint b = s >> 1;
        uint half_off = (s & 1) * 8;
        device const float* xb = xb_base + b * 32 + half_off;

        float4 x_lo0 = *reinterpret_cast<device const float4*>(xb);
        float4 x_lo1 = *reinterpret_cast<device const float4*>(xb + 4);
        float4 x_hi0 = *reinterpret_cast<device const float4*>(xb + 16);
        float4 x_hi1 = *reinterpret_cast<device const float4*>(xb + 20);
        float sum_x = (x_lo0.x + x_lo0.y + x_lo0.z + x_lo0.w)
                    + (x_lo1.x + x_lo1.y + x_lo1.z + x_lo1.w)
                    + (x_hi0.x + x_hi0.y + x_hi0.z + x_hi0.w)
                    + (x_hi1.x + x_hi1.y + x_hi1.z + x_hi1.w);

        uint2 q0 = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(r0_blocks[b].qs + half_off)));
        float dot0 = dot(float4(as_type<uchar4>(q0.x & 0x0F0F0F0Fu)), x_lo0)
                   + dot(float4(as_type<uchar4>(q0.y & 0x0F0F0F0Fu)), x_lo1)
                   + dot(float4(as_type<uchar4>((q0.x >> 4) & 0x0F0F0F0Fu)), x_hi0)
                   + dot(float4(as_type<uchar4>((q0.y >> 4) & 0x0F0F0F0Fu)), x_hi1);
        acc0 += float(r0_blocks[b].d) * (dot0 - 8.0f * sum_x);

        if (has_row1) {
            uint2 q1 = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(r1_blocks[b].qs + half_off)));
            float dot1 = dot(float4(as_type<uchar4>(q1.x & 0x0F0F0F0Fu)), x_lo0)
                       + dot(float4(as_type<uchar4>(q1.y & 0x0F0F0F0Fu)), x_lo1)
                       + dot(float4(as_type<uchar4>((q1.x >> 4) & 0x0F0F0F0Fu)), x_hi0)
                       + dot(float4(as_type<uchar4>((q1.y >> 4) & 0x0F0F0F0Fu)), x_hi1);
            acc1 += float(r1_blocks[b].d) * (dot1 - 8.0f * sum_x);
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        if (has_row1) acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

struct block_q8_0 {
    half d;
    int8_t qs[32];
};

kernel void gemv_q8_0(device const block_q8_0* w [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant PackedGemvParams& p [[buffer(3)]],
                      device const float* bias [[buffer(4)]],
                      device const float* res [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 32;
    uint n_sub = n_blocks * 4;
    device const block_q8_0* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q8_0* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint s = lane; s < n_sub; s += 32) {
        uint b = s >> 2;
        uint j = (s & 3) * 8;
        device const float* xb = xb_base + b * 32 + j;
        float4 x0 = *reinterpret_cast<device const float4*>(xb);
        float4 x1 = *reinterpret_cast<device const float4*>(xb + 4);
        uint2 q0 = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(r0_blocks[b].qs + j)));
        acc0 += float(r0_blocks[b].d) * (dot(float4(as_type<char4>(q0.x)), x0) + dot(float4(as_type<char4>(q0.y)), x1));
        if (has_row1) {
            uint2 q1 = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(r1_blocks[b].qs + j)));
            acc1 += float(r1_blocks[b].d) * (dot(float4(as_type<char4>(q1.x)), x0) + dot(float4(as_type<char4>(q1.y)), x1));
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_bf16(device const ushort* w [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant PackedGemvParams& p [[buffer(3)]],
                      device const float* bias [[buffer(4)]],
                      device const float* res [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    device const ushort* r0_w = w + (ulong)row0 * p.cols;
    device const ushort* r1_w = has_row1 ? (w + (ulong)row1 * p.cols) : r0_w;
    device const float* xb = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint col_rem = p.cols % 8;
    uint col_aligned = p.cols - col_rem;

    for (uint i = lane * 8; i < col_aligned; i += 32 * 8) {
        ushort4 w0_0 = *reinterpret_cast<device const ushort4*>(r0_w + i);
        ushort4 w0_1 = *reinterpret_cast<device const ushort4*>(r0_w + i + 4);
        ushort4 w1_0 = *reinterpret_cast<device const ushort4*>(r1_w + i);
        ushort4 w1_1 = *reinterpret_cast<device const ushort4*>(r1_w + i + 4);

        float4 xb0 = *reinterpret_cast<device const float4*>(xb + i);
        float4 xb1 = *reinterpret_cast<device const float4*>(xb + i + 4);

        float4 val0_0 = as_type<float4>(uint4(w0_0) << 16);
        float4 val0_1 = as_type<float4>(uint4(w0_1) << 16);
        acc0 += dot(val0_0, xb0) + dot(val0_1, xb1);

        if (has_row1) {
            float4 val1_0 = as_type<float4>(uint4(w1_0) << 16);
            float4 val1_1 = as_type<float4>(uint4(w1_1) << 16);
            acc1 += dot(val1_0, xb0) + dot(val1_1, xb1);
        }
    }
    for (uint i = col_aligned + lane; i < p.cols; i += 32) {
        uint u0 = uint(r0_w[i]) << 16;
        acc0 += as_type<float>(u0) * xb[i];
        if (has_row1) {
            uint u1 = uint(r1_w[i]) << 16;
            acc1 += as_type<float>(u1) * xb[i];
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        if (has_row1) acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_f16(device const half* w [[buffer(0)]],
                     device const float* x [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant PackedGemvParams& p [[buffer(3)]],
                     device const float* bias [[buffer(4)]],
                     device const float* res [[buffer(5)]],
                     uint2 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    device const half* r0_w = w + (ulong)row0 * p.cols;
    device const half* r1_w = has_row1 ? (w + (ulong)row1 * p.cols) : r0_w;
    device const float* xb = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint col_rem = p.cols % 8;
    uint col_aligned = p.cols - col_rem;

    for (uint i = lane * 8; i < col_aligned; i += 32 * 8) {
        half4 w0_0 = *reinterpret_cast<device const half4*>(r0_w + i);
        half4 w0_1 = *reinterpret_cast<device const half4*>(r0_w + i + 4);
        half4 w1_0 = *reinterpret_cast<device const half4*>(r1_w + i);
        half4 w1_1 = *reinterpret_cast<device const half4*>(r1_w + i + 4);

        float4 xb0 = *reinterpret_cast<device const float4*>(xb + i);
        float4 xb1 = *reinterpret_cast<device const float4*>(xb + i + 4);

        acc0 += dot(float4(w0_0), xb0) + dot(float4(w0_1), xb1);
        if (has_row1) {
            acc1 += dot(float4(w1_0), xb0) + dot(float4(w1_1), xb1);
        }
    }
    for (uint i = col_aligned + lane; i < p.cols; i += 32) {
        acc0 += float(r0_w[i]) * xb[i];
        if (has_row1) {
            acc1 += float(r1_w[i]) * xb[i];
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}


struct block_q4_k {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

inline float2 k4_scale_min(uint j, thread const uint8_t* s) {
    if (j < 4) {
        return float2(float(s[j] & 63), float(s[j + 4] & 63));
    } else {
        float sc = float((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        float m = float((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
        return float2(sc, m);
    }
}

inline float2 k4_scale_min(uint j, device const uint8_t* s) {
    if (j < 4) {
        return float2(float(s[j] & 63), float(s[j + 4] & 63));
    } else {
        float sc = float((s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4));
        float m = float((s[j + 4] >> 4) | ((s[j] >> 6) << 4));
        return float2(sc, m);
    }
}

kernel void gemv_q4_k(device const block_q4_k* w [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant PackedGemvParams& p [[buffer(3)]],
                      device const float* bias [[buffer(4)]],
                      device const float* res [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 256;
    uint n_sub = n_blocks * 16;
    device const block_q4_k* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q4_k* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint s = lane; s < n_sub; s += 32) {
        uint b = s >> 4;
        uint g = (s >> 2) & 3;
        uint sub = s & 3;
        uint q_off = g * 32 + sub * 8;
        device const float* xb_g = xb_base + (ulong)(b * 256 + g * 64 + sub * 8);

        float4 x_lo0 = *reinterpret_cast<device const float4*>(xb_g);
        float4 x_lo1 = *reinterpret_cast<device const float4*>(xb_g + 4);
        float4 x_hi0 = *reinterpret_cast<device const float4*>(xb_g + 32);
        float4 x_hi1 = *reinterpret_cast<device const float4*>(xb_g + 36);
        float sum_x_lo = (x_lo0.x + x_lo0.y + x_lo0.z + x_lo0.w)
                       + (x_lo1.x + x_lo1.y + x_lo1.z + x_lo1.w);
        float sum_x_hi = (x_hi0.x + x_hi0.y + x_hi0.z + x_hi0.w)
                       + (x_hi1.x + x_hi1.y + x_hi1.z + x_hi1.w);

        device const block_q4_k& blk0 = r0_blocks[b];
        float2 sm0_0 = k4_scale_min(2 * g, blk0.scales);
        float2 sm1_0 = k4_scale_min(2 * g + 1, blk0.scales);
        uint2 q0 = uint2(*reinterpret_cast<device const packed_uint2*>(blk0.qs + q_off));
        float dot_lo0 = dot(float4(as_type<uchar4>(q0.x & 0x0F0F0F0Fu)), x_lo0)
                      + dot(float4(as_type<uchar4>(q0.y & 0x0F0F0F0Fu)), x_lo1);
        float dot_hi0 = dot(float4(as_type<uchar4>((q0.x >> 4) & 0x0F0F0F0Fu)), x_hi0)
                      + dot(float4(as_type<uchar4>((q0.y >> 4) & 0x0F0F0F0Fu)), x_hi1);
        acc0 += float(blk0.d) * (sm0_0.x * dot_lo0 + sm1_0.x * dot_hi0)
              - float(blk0.dmin) * (sm0_0.y * sum_x_lo + sm1_0.y * sum_x_hi);

        if (has_row1) {
            device const block_q4_k& blk1 = r1_blocks[b];
            float2 sm0_1 = k4_scale_min(2 * g, blk1.scales);
            float2 sm1_1 = k4_scale_min(2 * g + 1, blk1.scales);
            uint2 q1 = uint2(*reinterpret_cast<device const packed_uint2*>(blk1.qs + q_off));
            float dot_lo1 = dot(float4(as_type<uchar4>(q1.x & 0x0F0F0F0Fu)), x_lo0)
                          + dot(float4(as_type<uchar4>(q1.y & 0x0F0F0F0Fu)), x_lo1);
            float dot_hi1 = dot(float4(as_type<uchar4>((q1.x >> 4) & 0x0F0F0F0Fu)), x_hi0)
                          + dot(float4(as_type<uchar4>((q1.y >> 4) & 0x0F0F0F0Fu)), x_hi1);
            acc1 += float(blk1.d) * (sm0_1.x * dot_lo1 + sm1_1.x * dot_hi1)
                  - float(blk1.dmin) * (sm0_1.y * sum_x_lo + sm1_1.y * sum_x_hi);
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

struct block_q2_k {
    uint8_t scales[16];
    uint8_t qs[64];
    half d;
    half dmin;
};

kernel void gemv_q2_k(device const block_q2_k* w [[buffer(0)]],
                      device const float* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant PackedGemvParams& p [[buffer(3)]],
                      device const float* bias [[buffer(4)]],
                      device const float* res [[buffer(5)]],
                      uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 256;
    device const block_q2_k* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q2_k* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // Each thread in SIMDgroup processes a subset of the 256-element block across all blocks in the row
    for (uint b = 0; b < n_blocks; ++b) {
        device const block_q2_k& blk0 = r0_blocks[b];
        device const block_q2_k& blk1 = has_row1 ? r1_blocks[b] : blk0;
        device const float* xb = xb_base + (ulong)b * 256;

        float d0 = float(blk0.d);
        float min0 = float(blk0.dmin);
        float d1 = float(blk1.d);
        float min1 = float(blk1.dmin);

        // 16 sub-blocks of 16 values = 256 values total.
        // lane (0..31) splits across the 16 sub-blocks: sub = lane / 2 (0..15), half = lane % 2 (8 values each).
        uint is = lane >> 1; // 0..15
        uint half_idx = lane & 1; // 0 or 1

        uint sc0 = blk0.scales[is];
        float dl0 = d0 * float(sc0 & 0x0F);
        float ml0 = min0 * float(sc0 >> 4);
        uint sc1 = blk1.scales[is];
        float dl1 = d1 * float(sc1 & 0x0F);
        float ml1 = min1 * float(sc1 >> 4);

        uint n_part = is >> 3;
        uint shift = (is & 6);
        uint half_sub = is & 1;

        uint q_off = n_part * 32 + half_sub * 16 + half_idx * 8;
        device const float* x_sub = xb + is * 16 + half_idx * 8;

        float4 x0 = *reinterpret_cast<device const float4*>(x_sub);
        float4 x1 = *reinterpret_cast<device const float4*>(x_sub + 4);
        float sum_x = (x0.x + x0.y + x0.z + x0.w) + (x1.x + x1.y + x1.z + x1.w);

        uint2 q0 = uint2(*reinterpret_cast<device const packed_uint2*>(blk0.qs + q_off));
        float dot0 = dot(float4(as_type<uchar4>((q0.x >> shift) & 0x03030303u)), x0)
                   + dot(float4(as_type<uchar4>((q0.y >> shift) & 0x03030303u)), x1);
        acc0 += dl0 * dot0 - ml0 * sum_x;

        if (has_row1) {
            uint2 q1 = uint2(*reinterpret_cast<device const packed_uint2*>(blk1.qs + q_off));
            float dot1 = dot(float4(as_type<uchar4>((q1.x >> shift) & 0x03030303u)), x0)
                       + dot(float4(as_type<uchar4>((q1.y >> shift) & 0x03030303u)), x1);
            acc1 += dl1 * dot1 - ml1 * sum_x;
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}


kernel void gemv_tq2_0(device const uint8_t* w [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device float* y [[buffer(2)]],
                       constant PackedGemvParams& p [[buffer(3)]],
                       device const float* bias [[buffer(4)]],
                       device const float* res [[buffer(5)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = tg.x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint bytes = p.cols / 4;
    device const uint8_t* r0_w = w + (ulong)row0 * bytes;
    device const uint8_t* r1_w = has_row1 ? (w + (ulong)row1 * bytes) : r0_w;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint b = lane; b < bytes; b += 32) {
        uint8_t byte0 = r0_w[b];
        uint8_t byte1 = r1_w[b];
        float4 xb = *reinterpret_cast<device const float4*>(xb_base + b * 4);
        for (uint j = 0; j < 4; ++j) {
            int code0 = int((byte0 >> (j * 2)) & 0x03);
            int code1 = int((byte1 >> (j * 2)) & 0x03);
            float xj = xb[j];
            acc0 += float(code0 - 1) * xj;
            acc1 += float(code1 - 1) * xj;
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);
    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

enum : int {
    FMT_BF16 = 0,
    FMT_F16 = 1,
    FMT_Q4_0 = 2,
    FMT_Q8_0 = 3,
};

template <int FORMAT, int EPI>
kernel void gemm_batched_packed(device const float* x [[buffer(0)]],
                                device const void* w [[buffer(1)]],
                                device const float* bias [[buffer(2)]],
                                device const float* res [[buffer(3)]],
                                device float* out [[buffer(4)]],
                                constant GemmParams& p [[buffer(5)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[32 * 8];  // [row][k]
    threadgroup float ws[8 * 32];  // [k][col]
    threadgroup float cs[32 * 32];
    const uint m0 = tg.y * 32, n0 = tg.x * 32;
    const uint sm = (sg / 2) * 16, sn = (sg % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    device const ushort* w_bf16 = (device const ushort*)w;
    device const half* w_f16 = (device const half*)w;
    device const block_q4_0* w_q4 = (device const block_q4_0*)w;
    device const block_q8_0* w_q8 = (device const block_q8_0*)w;
    const uint q4_n_blocks = p.k / 32;
    const uint q8_n_blocks = p.k / 32;

    for (uint k0 = 0; k0 < p.k; k0 += 8) {
        for (uint e = tid; e < 256; e += 128) {
            uint r = e / 8, kk = e % 8, gk = k0 + kk;
            xs[e] = (m0 + r < p.m && gk < p.k) ? x[(ulong)(m0 + r) * p.k + gk] : 0.0f;

            uint col = n0 + r;
            float val = 0.0f;
            if (col < p.n && gk < p.k) {
                if (FORMAT == FMT_BF16) {
                    ushort u = w_bf16[(ulong)col * p.k + gk];
                    val = as_type<float>(uint(u) << 16);
                } else if (FORMAT == FMT_F16) {
                    val = float(w_f16[(ulong)col * p.k + gk]);
                } else if (FORMAT == FMT_Q8_0) {
                    device const block_q8_0* blk = w_q8 + (ulong)col * q8_n_blocks + (gk / 32);
                    val = float(blk->d) * float(blk->qs[gk % 32]);
                } else if (FORMAT == FMT_Q4_0) {
                    device const block_q4_0* blk = w_q4 + (ulong)col * q4_n_blocks + (gk / 32);
                    uint off = gk % 32;
                    int q = (off < 16) ? (int(blk->qs[off] & 0x0F) - 8) : (int(blk->qs[off - 16] >> 4) - 8);
                    val = float(blk->d) * float(q);
                }
            }
            ws[kk * 32 + r] = val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 a[2], b[2];
        for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], xs + (sm + 8 * i) * 8, 8);
        for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], ws + sn + 8 * j, 32);
        for (uint i = 0; i < 2; ++i)
            for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 2; ++j) simdgroup_store(acc[i][j], cs + (sm + 8 * i) * 32 + sn + 8 * j, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 1024; e += 128) {
        uint r = m0 + e / 32, c = n0 + e % 32;
        if (r < p.m && c < p.n) out[(ulong)r * p.n + c] = epilogue<EPI>(cs[e], r, c, p, bias, res);
    }
}

#define GEMM_BATCHED_INSTANCES(fmt_name, fmt_val) \
    template [[host_name("gemm_batched_" fmt_name "_none")]] [[kernel]] decltype(gemm_batched_packed<fmt_val, EPI_NONE>) gemm_batched_packed<fmt_val, EPI_NONE>; \
    template [[host_name("gemm_batched_" fmt_name "_gelu")]] [[kernel]] decltype(gemm_batched_packed<fmt_val, EPI_GELU>) gemm_batched_packed<fmt_val, EPI_GELU>; \
    template [[host_name("gemm_batched_" fmt_name "_residual")]] [[kernel]] decltype(gemm_batched_packed<fmt_val, EPI_RESIDUAL>) gemm_batched_packed<fmt_val, EPI_RESIDUAL>;

GEMM_BATCHED_INSTANCES("bf16", FMT_BF16)
GEMM_BATCHED_INSTANCES("f16", FMT_F16)
GEMM_BATCHED_INSTANCES("q4_0", FMT_Q4_0)
GEMM_BATCHED_INSTANCES("q8_0", FMT_Q8_0)

struct SliceParams {
    uint rows;
    uint in_cols;
    uint start;
    uint out_cols;
};

struct GraphRopeParams2 {
    uint n_heads;
    uint head_dim;
    uint pairs;
    uint pair_stride;
    uint pair_gap;
    uint rotary_dim;
    float mscale;
    uint n_seq;
};

inline void rope_elem(device const float* in, device float* out, device const uint* pos,
                      device const float* inv_freq, uint n_heads, uint head_dim, uint rotary_dim,
                      uint pair_stride, float mscale, uint gid) {
    uint d = gid % head_dim;
    uint rem = gid / head_dim;
    uint head = rem % n_heads;
    uint seq = rem / n_heads;

    device const float* in_h = in + (ulong)seq * (n_heads * head_dim) + (ulong)head * head_dim;
    device float* out_h = out + (ulong)seq * (n_heads * head_dim) + (ulong)head * head_dim;

    if (d < rotary_dim) {
        if (pair_stride == 2) {
            uint pair_idx = d / 2;
            float freq = inv_freq[pair_idx];
            if (freq == 0.0f) {
                out_h[d] = in_h[d] * mscale;
            } else {
                float theta = float(pos[seq]) * freq;
                float cos_th = cos(theta) * mscale;
                float sin_th = sin(theta) * mscale;
                float a = in_h[pair_idx * 2];
                float b = in_h[pair_idx * 2 + 1];
                out_h[d] = (d % 2 == 0) ? (a * cos_th - b * sin_th) : (b * cos_th + a * sin_th);
            }
        } else {
            uint half_dim = rotary_dim / 2;
            if (d < half_dim) {
                float freq = inv_freq[d];
                if (freq == 0.0f) {
                    out_h[d] = in_h[d] * mscale;
                    out_h[d + half_dim] = in_h[d + half_dim] * mscale;
                } else {
                    float theta = float(pos[seq]) * freq;
                    float cos_th = cos(theta) * mscale;
                    float sin_th = sin(theta) * mscale;
                    float a = in_h[d];
                    float b = in_h[d + half_dim];
                    out_h[d] = a * cos_th - b * sin_th;
                    out_h[d + half_dim] = b * cos_th + a * sin_th;
                }
            }
        }
    } else {
        out_h[d] = in_h[d];
    }
}

kernel void rope_general_f32(device const float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             device const uint* pos [[buffer(2)]],
                             device const float* inv_freq [[buffer(3)]],
                             constant GraphRopeParams2& p [[buffer(4)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= p.n_seq * p.n_heads * p.head_dim) return;
    rope_elem(in, out, pos, inv_freq, p.n_heads, p.head_dim, p.rotary_dim, p.pair_stride, p.mscale, gid);
}

// Horizontal batch of up to 4 Rope segments sharing pos/inv_freq/head_dim/rotary_dim/pair_stride/mscale/n_seq;
// each has its own input, output and head count. Thread ranges are consecutive per segment.
struct RopeBatchParams {
    uint n_heads[4];
    uint nseg;
    uint head_dim;
    uint rotary_dim;
    uint pair_stride;
    float mscale;
    uint n_seq;
};
kernel void rope_batch_f32(device const float* in0 [[buffer(0)]], device float* out0 [[buffer(1)]],
                           device const float* in1 [[buffer(2)]], device float* out1 [[buffer(3)]],
                           device const float* in2 [[buffer(4)]], device float* out2 [[buffer(5)]],
                           device const float* in3 [[buffer(6)]], device float* out3 [[buffer(7)]],
                           device const uint* pos [[buffer(8)]],
                           device const float* inv_freq [[buffer(9)]],
                           constant RopeBatchParams& p [[buffer(10)]],
                           uint gid [[thread_position_in_grid]]) {
    device const float* ins[4] = {in0, in1, in2, in3};
    device float* outs[4] = {out0, out1, out2, out3};
    uint g = gid;
    for (uint s = 0; s < p.nseg; s++) {
        uint total = p.n_seq * p.n_heads[s] * p.head_dim;
        if (g < total) {
            rope_elem(ins[s], outs[s], pos, inv_freq, p.n_heads[s], p.head_dim, p.rotary_dim,
                      p.pair_stride, p.mscale, g);
            return;
        }
        g -= total;
    }
}

struct KvAppendParams {
    uint m;
    uint kv_dim;
    uint stride;
    uint kv_base;    // stored row of the first written row (`crate::kv` module docs)
    uint ring_pages; // ring length of a paged windowed state, 0 otherwise
    uint paged;      // the state operands are `KvDesc` bindings
};

// A paged KV state's binding (`kv::KvView::metal_desc`): header, then the class page table (one
// page id per logical 64-row page). `slabs[pg >> 8]` is the base of the name's slab holding page `pg`.
struct KvDesc {
    uint ring_pages; // host-read; kernels take it from their params
    uint evicted;    // host-read
    uint n_pages;    // host-read
    uint pad;
    device uchar* device* slabs;
    ulong pad2;
};

// Stored row `t` of a KV state with `stride` elements per row: the contiguous buffer `kv`, or,
// when `paged`, the page `table[(t >> 6) % ring]` of the `KvDesc` bound at `kv` (`crate::kv`).
template <typename T>
inline device T* kv_row(device T* kv, uint paged, uint ring, uint t, ulong stride) {
    if (paged == 0) return kv + (ulong)t * stride;
    device const KvDesc* d = (device const KvDesc*)kv;
    uint lp = t >> 6;
    if (ring != 0) lp %= ring;
    const uint pg = ((device const uint*)(d + 1))[lp];
    return (device T*)(d->slabs[pg >> 8]) + (ulong)((pg & 255u) * 64u + (t & 63u)) * stride;
}

// KV cache element type: float, or half (round-to-nearest-even; |x| > 65504 saturates instead of becoming inf,
// NaN is kept).
inline float kv_store(float x, float) { return x; }
inline half kv_store(float x, half) { return isnan(x) ? half(x) : half(clamp(x, -65504.0f, 65504.0f)); }

template <typename T>
kernel void kv_append_t(device const float* k_in [[buffer(0)]],
                        device const float* v_in [[buffer(1)]],
                        device T* k_state [[buffer(2)]],
                        device T* v_state [[buffer(3)]],
                        constant KvAppendParams& p [[buffer(4)]],
                        uint gid [[thread_position_in_grid]]) {
    uint total = p.m * p.kv_dim;
    if (gid >= total) return;
    uint tok = gid / p.kv_dim;
    uint d = gid % p.kv_dim;
    device T* kr = kv_row(k_state, p.paged, p.ring_pages, p.kv_base + tok, p.stride);
    device T* vr = kv_row(v_state, p.paged, p.ring_pages, p.kv_base + tok, p.stride);
    kr[d] = kv_store(k_in[gid], T(0));
    vr[d] = kv_store(v_in[gid], T(0));
}

typedef decltype(kv_append_t<float>) kv_append_k;
typedef decltype(kv_append_t<half>) kv_append_kh;
template [[host_name("kv_append_batched")]] kernel kv_append_k kv_append_t<float>;
template [[host_name("kv_append_batched_f16")]] kernel kv_append_kh kv_append_t<half>;

struct AttnCachedParams {
    uint n_heads;
    uint head_dim;
    uint stride;
    uint group;
    float scale;
    uint window;
    float softcap;
    uint start_pos;
    uint m;
    uint has_vis;
    uint n_splits; // > 1: this launch covers one `chunk`-key slice of the row's keys (grid z = slice)
    uint chunk;
    uint write_kv;
    uint rotary_dim;
    uint pair_stride;
    float mscale;
    uint ring_pages; // ring length of paged windowed states, 0 otherwise
    uint kv_base;    // stored rows before this step (`start_pos` unless Tier-4 evicted pages)
    uint paged;      // the k/v state operands are `KvDesc` bindings
};

template <uint NJ, typename T>
inline void attn_cached_f32_impl(device const float* q,
                                 device T* kc,
                                 device T* vc,
                                 device float* out,
                                 constant AttnCachedParams& p,
                                 device const uint* vis,
                                 device const float* k_in,
                                 device const float* v_in,
                                 uint2 tg,
                                 uint sg,
                                 uint lane,
                                 uint nsg,
                                 threadgroup float* tg_m,
                                 threadgroup float* tg_l,
                                 threadgroup float* tg_acc) {
    const uint hd = p.head_dim;
    const uint row = NJ * 32;
    const uint head = tg.x;
    const uint q_tok = tg.y;
    if (head >= p.n_heads || q_tok >= p.m) return;

    device const float* qh = q + ((ulong)q_tok * p.n_heads + head) * hd;
    const uint kv_off = (head / p.group) * hd;
    if (p.write_kv != 0 && (head % p.group) == 0 && q_tok == 0) {
        const uint tid = sg * 32 + lane;
        device T* kd = kv_row(kc, p.paged, p.ring_pages, p.kv_base, p.stride) + kv_off;
        device T* vd = kv_row(vc, p.paged, p.ring_pages, p.kv_base, p.stride) + kv_off;
        for (uint d = tid; d < hd; d += nsg * 32) {
            kd[d] = kv_store(k_in[kv_off + d], T(0));
            vd[d] = kv_store(v_in[kv_off + d], T(0));
        }
    }
    // Stored-row space: query row `q_tok` is row `kv_base + q_tok`; `vis` holds positions.
    const uint cur_pos = p.kv_base + q_tok;
    const uint len = p.has_vis ? vis[q_tok] - (p.start_pos - p.kv_base) : cur_pos + 1;
    float qr[NJ];
    float acc[NJ];
    for (uint j = 0; j < NJ; ++j) {
        uint d = lane + 32 * j;
        qr[j] = d < hd ? qh[d] : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -FLT_MAX;
    float l = 0.0f;
    const uint lo = (p.window > 0 && cur_pos + 1 > p.window) ? (cur_pos + 1 - p.window) : 0;
    const uint hi = len;
    for (uint t = lo + sg; t < hi; t += nsg) {
        // The fused write's own row comes from k_in / v_in (other heads' threadgroups may not see the store).
        const bool cur_in = p.write_kv != 0 && t == p.kv_base;
        device const T* k = kv_row(kc, p.paged, p.ring_pages, t, p.stride) + kv_off;
        device const T* v = kv_row(vc, p.paged, p.ring_pages, t, p.stride) + kv_off;
        device const float* ki = k_in + kv_off;
        device const float* vi = v_in + kv_off;
        float s = 0.0f;
        for (uint j = 0; j < NJ; ++j) {
            uint d = lane + 32 * j;
            if (d < hd) s += qr[j] * (cur_in ? ki[d] : float(k[d]));
        }
        s = simd_sum(s) * p.scale;
        if (p.softcap > 0.0f) {
            s = tanh(s / p.softcap) * p.softcap;
        }
        float m_new = max(m, s);
        float corr = exp(m - m_new);
        float e = exp(s - m_new);
        l = l * corr + e;
        for (uint j = 0; j < NJ; ++j) {
            uint d = lane + 32 * j;
            if (d < hd) acc[j] = acc[j] * corr + e * (cur_in ? vi[d] : float(v[d]));
        }
        m = m_new;
    }
    if (lane == 0) {
        tg_m[sg] = m;
        tg_l[sg] = l;
    }
    for (uint j = 0; j < NJ; ++j) {
        uint d = lane + 32 * j;
        if (d < hd) tg_acc[sg * row + d] = acc[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = -FLT_MAX;
    for (uint i = 0; i < nsg; ++i) gm = max(gm, tg_m[i]);
    float gl = 0.0f;
    for (uint i = 0; i < nsg; ++i) gl += tg_l[i] * exp(tg_m[i] - gm);
    for (uint d = sg * 32 + lane; d < hd; d += nsg * 32) {
        float o = 0.0f;
        for (uint i = 0; i < nsg; ++i) o += tg_acc[i * row + d] * exp(tg_m[i] - gm);
        out[((ulong)q_tok * p.n_heads + head) * hd + d] = o / gl;
    }
}

template <uint NJ, typename T>
kernel void flash_attn_cached_t(device const float* q [[buffer(0)]],
                                device T* kc [[buffer(1)]],
                                device T* vc [[buffer(2)]],
                                device float* out [[buffer(3)]],
                                constant AttnCachedParams& p [[buffer(4)]],
                                device const uint* vis [[buffer(5)]],
                                device const float* k_in [[buffer(6)]],
                                device const float* v_in [[buffer(7)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float tg_m[ATTN_MAX_SG];
    threadgroup float tg_l[ATTN_MAX_SG];
    threadgroup float tg_acc[ATTN_MAX_SG * NJ * 32];
    attn_cached_f32_impl<NJ, T>(q, kc, vc, out, p, vis, k_in, v_in, tg, sg, lane, nsg, tg_m, tg_l, tg_acc);
}

typedef decltype(flash_attn_cached_t<1, float>) flash_attn_cached_k;
typedef decltype(flash_attn_cached_t<1, half>) flash_attn_cached_kh;
template [[host_name("flash_attn_cached_f32_1")]] kernel flash_attn_cached_k flash_attn_cached_t<1, float>;
template [[host_name("flash_attn_cached_f32_2")]] kernel flash_attn_cached_k flash_attn_cached_t<2, float>;
template [[host_name("flash_attn_cached_f32_4")]] kernel flash_attn_cached_k flash_attn_cached_t<4, float>;
template [[host_name("flash_attn_cached_f32_8")]] kernel flash_attn_cached_k flash_attn_cached_t<8, float>;
template [[host_name("flash_attn_cached_f32")]] kernel flash_attn_cached_k flash_attn_cached_t<16, float>;
template [[host_name("flash_attn_cached_f16_1")]] kernel flash_attn_cached_kh flash_attn_cached_t<1, half>;
template [[host_name("flash_attn_cached_f16_2")]] kernel flash_attn_cached_kh flash_attn_cached_t<2, half>;
template [[host_name("flash_attn_cached_f16_4")]] kernel flash_attn_cached_kh flash_attn_cached_t<4, half>;
template [[host_name("flash_attn_cached_f16_8")]] kernel flash_attn_cached_kh flash_attn_cached_t<8, half>;
template [[host_name("flash_attn_cached_f16")]] kernel flash_attn_cached_kh flash_attn_cached_t<16, half>;
kernel void attn_cached_f32(device const float* q [[buffer(0)]],
                            device float* kc [[buffer(1)]],
                            device float* vc [[buffer(2)]],
                            device float* out [[buffer(3)]],
                            constant AttnCachedParams& p [[buffer(4)]],
                            device const uint* vis [[buffer(5)]],
                            device const float* k_in [[buffer(6)]],
                            device const float* v_in [[buffer(7)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float tg_m[ATTN_MAX_SG];
    threadgroup float tg_l[ATTN_MAX_SG];
    threadgroup float tg_acc[ATTN_MAX_SG * ATTN_MAX_PER_LANE * 32];
    attn_cached_f32_impl<ATTN_MAX_PER_LANE, float>(q, kc, vc, out, p, vis, k_in, v_in, tg, sg, lane, nsg, tg_m, tg_l, tg_acc);
}

kernel void attn_cached_merge_f32(device const float* part [[buffer(0)]],
                                  device float* out [[buffer(1)]],
                                  constant AttnCachedParams& p [[buffer(2)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint2 ntg [[threads_per_threadgroup]]) {
    const uint hd = p.head_dim, head = tg.x, q_tok = tg.y;
    if (head >= p.n_heads || q_tok >= p.m) return;
    device const float* base = part + ((ulong)q_tok * p.n_heads + head) * p.n_splits * (hd + 2);
    float gm = -FLT_MAX;
    for (uint s = 0; s < p.n_splits; ++s) gm = max(gm, base[s * (hd + 2)]);
    float gl = 0.0f;
    for (uint s = 0; s < p.n_splits; ++s) gl += base[s * (hd + 2) + 1] * exp(base[s * (hd + 2)] - gm);
    float inv_gl = 1.0f / gl;
    for (uint d = tid; d < hd; d += ntg.x) {
        float o = 0.0f;
        for (uint s = 0; s < p.n_splits; ++s) o += base[s * (hd + 2) + 2 + d] * exp(base[s * (hd + 2)] - gm);
        out[((ulong)q_tok * p.n_heads + head) * hd + d] = o * inv_gl;
    }
}

// Decode attention for grouped KV heads: grid (kv_heads * ceil(group / ATTN_HPT), 1, slices). A threadgroup owns up to
// ATTN_HPT query heads that share one KV head and one `p.chunk`-key slice, so each key and value is loaded
// once for all of them (the per-head kernel above re-reads them `group` times). Partials have the layout of
// `attn_cached_split_f32`; `attn_cached_merge_f32` combines them.
constant uint ATTN_HPT = 4;
inline float attn_rope_val(device const float* xh,
                           uint d,
                           uint pos,
                           device const float* inv_freq,
                           uint rotary_dim,
                           uint pair_stride,
                           float mscale) {
    if (d >= rotary_dim) return xh[d];
    if (pair_stride == 2) {
        uint pair = d >> 1;
        float freq = inv_freq[pair];
        if (freq == 0.0f) return xh[d] * mscale;
        float theta = (float)pos * freq;
        float c = cos(theta) * mscale;
        float s = sin(theta) * mscale;
        float a = xh[pair * 2];
        float b = xh[pair * 2 + 1];
        return (d & 1) == 0 ? (a * c - b * s) : (b * c + a * s);
    } else {
        uint half_d = rotary_dim >> 1;
        uint pair = d < half_d ? d : (d - half_d);
        float freq = inv_freq[pair];
        if (freq == 0.0f) return xh[d] * mscale;
        float theta = (float)pos * freq;
        float c = cos(theta) * mscale;
        float s = sin(theta) * mscale;
        float a = xh[pair];
        float b = xh[pair + half_d];
        return d < half_d ? (a * c - b * s) : (b * c + a * s);
    }
}

template <uint NJ, typename T>
kernel void attn_cached_gqa_split_t(device const float* q [[buffer(0)]],
                                      device T* kc [[buffer(1)]],
                                      device T* vc [[buffer(2)]],
                                      device float* part [[buffer(3)]],
                                      constant AttnCachedParams& p [[buffer(4)]],
                                      device const float* k_in [[buffer(5)]],
                                      device const float* v_in [[buffer(6)]],
                                      device const float* inv_freq [[buffer(7)]],
                                      uint3 tg3 [[threadgroup_position_in_grid]],
                                      uint sg [[simdgroup_index_in_threadgroup]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float tq[ATTN_HPT * NJ * 32];
    threadgroup float tg_m[ATTN_HPT * ATTN_MAX_SG];
    threadgroup float tg_l[ATTN_HPT * ATTN_MAX_SG];
    threadgroup float tg_red[ATTN_MAX_SG * NJ * 32];
    const uint hd = p.head_dim;
    const uint row = NJ * 32;
    const uint hgroups = (p.group + ATTN_HPT - 1) / ATTN_HPT;
    const uint kvh = tg3.x / hgroups;
    const uint head0 = kvh * p.group + (tg3.x % hgroups) * ATTN_HPT;
    const uint nh = min(ATTN_HPT, kvh * p.group + p.group - head0);
    const uint q_tok = 0;
    const uint split = tg3.z;
    const uint tid = sg * 32 + lane;
    for (uint i = tid; i < nh * row; i += nsg * 32) {
        const uint h = i / row, d = i % row;
        device const float* qh = q + ((ulong)q_tok * p.n_heads + head0 + h) * hd;
        tq[i] = d < hd
            ? (p.rotary_dim > 0
                ? attn_rope_val(qh, d, p.start_pos, inv_freq, p.rotary_dim, p.pair_stride, p.mscale)
                : qh[d])
            : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint kv_off = kvh * hd;
    if (p.write_kv != 0 && (tg3.x % hgroups) == 0 && split == 0) {
        device T* kd = kv_row(kc, p.paged, p.ring_pages, p.kv_base, p.stride) + kv_off;
        device T* vd = kv_row(vc, p.paged, p.ring_pages, p.kv_base, p.stride) + kv_off;
        for (uint d = tid; d < hd; d += nsg * 32) {
            kd[d] = kv_store(p.rotary_dim > 0
                ? attn_rope_val(k_in + kv_off, d, p.start_pos, inv_freq, p.rotary_dim, p.pair_stride, p.mscale)
                : k_in[kv_off + d], T(0));
            vd[d] = kv_store(v_in[kv_off + d], T(0));
        }
    }
    const uint cur_pos = p.kv_base + q_tok;
    const uint len = cur_pos + 1;
    uint lo = (p.window > 0 && cur_pos + 1 > p.window) ? (cur_pos + 1 - p.window) : 0;
    lo += split * p.chunk;
    const uint hi = min(len, lo + p.chunk);

    float acc[ATTN_HPT][NJ];
    float m[ATTN_HPT], l[ATTN_HPT];
    for (uint h = 0; h < ATTN_HPT; ++h) {
        m[h] = -FLT_MAX;
        l[h] = 0.0f;
        for (uint j = 0; j < NJ; ++j) acc[h][j] = 0.0f;
    }
    for (uint t = lo + sg; t < hi; t += nsg) {
        const bool cur_in = p.write_kv != 0 && t == p.kv_base;
        device const T* k = kv_row(kc, p.paged, p.ring_pages, t, p.stride) + kv_off;
        device const T* v = kv_row(vc, p.paged, p.ring_pages, t, p.stride) + kv_off;
        float kr[NJ];
        if (cur_in) {
            for (uint j = 0; j < NJ; ++j) {
                const uint d = lane + 32 * j;
                kr[j] = d < hd
                    ? (p.rotary_dim > 0
                        ? attn_rope_val(k_in + kv_off, d, p.start_pos, inv_freq, p.rotary_dim, p.pair_stride, p.mscale)
                        : k_in[kv_off + d])
                    : 0.0f;
            }
        } else {
            for (uint j = 0; j < NJ; ++j) {
                const uint d = lane + 32 * j;
                kr[j] = d < hd ? float(k[d]) : 0.0f;
            }
        }
        float vr[NJ];
        for (uint j = 0; j < NJ; ++j) {
            const uint d = lane + 32 * j;
            vr[j] = d < hd ? (cur_in ? v_in[kv_off + d] : float(v[d])) : 0.0f;
        }
        for (uint h = 0; h < ATTN_HPT; ++h) {
            if (h >= nh) break;
            float s = 0.0f;
            for (uint j = 0; j < NJ; ++j) s += tq[h * row + lane + 32 * j] * kr[j];
            s = simd_sum(s) * p.scale;
            if (p.softcap > 0.0f) s = tanh(s / p.softcap) * p.softcap;
            const float m_new = max(m[h], s);
            const float corr = exp(m[h] - m_new);
            const float e = exp(s - m_new);
            l[h] = l[h] * corr + e;
            for (uint j = 0; j < NJ; ++j) acc[h][j] = acc[h][j] * corr + e * vr[j];
            m[h] = m_new;
        }
    }
    for (uint h = 0; h < ATTN_HPT; ++h) {
        if (h >= nh) break;
        if (lane == 0) {
            tg_m[h * ATTN_MAX_SG + sg] = m[h];
            tg_l[h * ATTN_MAX_SG + sg] = l[h];
        }
        for (uint j = 0; j < NJ; ++j) {
            const uint d = lane + 32 * j;
            if (d < hd) tg_red[sg * row + d] = acc[h][j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float gm = -FLT_MAX;
        for (uint i = 0; i < nsg; ++i) gm = max(gm, tg_m[h * ATTN_MAX_SG + i]);
        float gl = 0.0f;
        for (uint i = 0; i < nsg; ++i) gl += tg_l[h * ATTN_MAX_SG + i] * exp(tg_m[h * ATTN_MAX_SG + i] - gm);
        if (p.n_splits == 1) {
            device float* out = part + ((ulong)q_tok * p.n_heads + head0 + h) * hd;
            float inv_gl = 1.0f / gl;
            for (uint d = tid; d < hd; d += nsg * 32) {
                float o = 0.0f;
                for (uint i = 0; i < nsg; ++i) o += tg_red[i * row + d] * exp(tg_m[h * ATTN_MAX_SG + i] - gm);
                out[d] = o * inv_gl;
            }
        } else {
            device float* out = part + (((ulong)q_tok * p.n_heads + head0 + h) * p.n_splits + split) * (hd + 2);
            if (tid == 0) {
                out[0] = gm;
                out[1] = gl;
            }
            for (uint d = tid; d < hd; d += nsg * 32) {
                float o = 0.0f;
                for (uint i = 0; i < nsg; ++i) o += tg_red[i * row + d] * exp(tg_m[h * ATTN_MAX_SG + i] - gm);
                out[2 + d] = o;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

typedef decltype(attn_cached_gqa_split_t<1, float>) attn_cached_gqa_split_k;
typedef decltype(attn_cached_gqa_split_t<1, half>) attn_cached_gqa_split_kh;
template [[host_name("attn_cached_gqa_split_f32_1")]] kernel attn_cached_gqa_split_k attn_cached_gqa_split_t<1, float>;
template [[host_name("attn_cached_gqa_split_f32_2")]] kernel attn_cached_gqa_split_k attn_cached_gqa_split_t<2, float>;
template [[host_name("attn_cached_gqa_split_f32_4")]] kernel attn_cached_gqa_split_k attn_cached_gqa_split_t<4, float>;
template [[host_name("attn_cached_gqa_split_f32_8")]] kernel attn_cached_gqa_split_k attn_cached_gqa_split_t<8, float>;
template [[host_name("attn_cached_gqa_split_f32_16")]] kernel attn_cached_gqa_split_k attn_cached_gqa_split_t<16, float>;
template [[host_name("attn_cached_gqa_split_f16_1")]] kernel attn_cached_gqa_split_kh attn_cached_gqa_split_t<1, half>;
template [[host_name("attn_cached_gqa_split_f16_2")]] kernel attn_cached_gqa_split_kh attn_cached_gqa_split_t<2, half>;
template [[host_name("attn_cached_gqa_split_f16_4")]] kernel attn_cached_gqa_split_kh attn_cached_gqa_split_t<4, half>;
template [[host_name("attn_cached_gqa_split_f16_8")]] kernel attn_cached_gqa_split_kh attn_cached_gqa_split_t<8, half>;
template [[host_name("attn_cached_gqa_split_f16_16")]] kernel attn_cached_gqa_split_kh attn_cached_gqa_split_t<16, half>;


struct Conv1dParams {
    uint n;
    uint channels;
    uint kernel_size;
    uint dilation;
    uint hist;
    uint has_state;
};

kernel void causal_conv1d_f32(device const float* x [[buffer(0)]],
                              device const float* w [[buffer(1)]],
                              device float* state [[buffer(2)]],
                              device float* out [[buffer(3)]],
                              constant Conv1dParams& p [[buffer(4)]],
                              uint ch [[thread_position_in_grid]]) {
    if (ch >= p.channels) return;
    uint k = p.kernel_size;
    uint d = p.dilation;
    uint hist = p.hist;
    device const float* w_ch = w + ch * k;

    for (uint i = 0; i < p.n; ++i) {
        float acc = 0.0f;
        for (uint tap = 0; tap < k; ++tap) {
            uint offset_step = (k - 1 - tap) * d;
            int sample_idx = int(hist + i) - int(offset_step);
            if (sample_idx >= 0) {
                float val;
                if (sample_idx < int(hist)) {
                    val = (p.has_state != 0) ? state[(ulong)sample_idx * p.channels + ch] : 0.0f;
                } else {
                    val = x[(ulong)(sample_idx - hist) * p.channels + ch];
                }
                acc += w_ch[tap] * val;
            }
        }
        out[(ulong)i * p.channels + ch] = acc;
    }

    if (p.has_state != 0) {
        for (uint j = 0; j < hist; ++j) {
            int idx = int(p.n + j);
            float val;
            if (idx < int(hist)) {
                val = state[(ulong)idx * p.channels + ch];
            } else {
                val = x[(ulong)(idx - hist) * p.channels + ch];
            }
            state[(ulong)j * p.channels + ch] = val;
        }
    }
}

struct DeltaRuleParams {
    uint n;
    uint key_heads;
    uint val_heads;
    uint key_dim;
    uint val_dim;
    uint rep;
};

// Gated delta rule, one threadgroup per (32 value columns, head): 4 simdgroups split the key rows of the
// state (`i = group + 4r`), lanes are the columns (coalesced state access), q/k are normalised once into
// threadgroup memory and the per-column sums are combined through it. Requires key_dim <= DELTA_MAX_KD.
constant uint DELTA_MAX_KD = 512;
kernel void delta_rule_f32(device const float* q [[buffer(0)]],
                           device const float* k [[buffer(1)]],
                           device const float* v [[buffer(2)]],
                           device const float* beta [[buffer(3)]],
                           device const float* log_decay [[buffer(4)]],
                           device float* state [[buffer(5)]],
                           device float* out [[buffer(6)]],
                           constant DeltaRuleParams& p [[buffer(7)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float ks[DELTA_MAX_KD];
    threadgroup float qs[DELTA_MAX_KD];
    threadgroup float red[8];
    threadgroup float part[128];
    const uint lane = tid & 31, g = tid >> 5;
    const uint j = tg.x * 32 + lane;
    const uint h = tg.y;
    const bool live = j < p.val_dim;
    const uint kh = p.key_heads, kd = p.key_dim, vd = p.val_dim, vh = p.val_heads;
    const uint k_h = h / p.rep;

    device float* ssm_state = state + (ulong)h * kd * vd;

    for (uint t = 0; t < p.n; ++t) {
        device const float* q_in = q + ((ulong)t * kh + k_h) * kd;
        device const float* k_in = k + ((ulong)t * kh + k_h) * kd;
        device const float* v_t = v + ((ulong)t * vh + h) * vd;
        const float b = beta[(ulong)t * vh + h];
        const float decay = exp(log_decay[(ulong)t * vh + h]);

        float qn = 0.0f, kn = 0.0f;
        for (uint i = tid; i < kd; i += 128) {
            qn += q_in[i] * q_in[i];
            kn += k_in[i] * k_in[i];
        }
        qn = simd_sum(qn);
        kn = simd_sum(kn);
        if (lane == 0) {
            red[g] = qn;
            red[4 + g] = kn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float q_norm = max(sqrt(red[0] + red[1] + red[2] + red[3]), 1e-12f);
        const float k_norm = max(sqrt(red[4] + red[5] + red[6] + red[7]), 1e-12f);
        const float q_scale = (1.0f / sqrt(float(kd))) / q_norm;
        const float k_scale = 1.0f / k_norm;
        for (uint i = tid; i < kd; i += 128) {
            ks[i] = k_in[i] * k_scale;
            qs[i] = q_in[i] * q_scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float kv = 0.0f;
        if (live) {
            for (uint i = g; i < kd; i += 4) kv += ssm_state[i * vd + j] * ks[i];
        }
        part[tid] = kv;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kv_j = part[lane] + part[32 + lane] + part[64 + lane] + part[96 + lane];
        const float delta_j = live ? b * (v_t[j] - decay * kv_j) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float o = 0.0f;
        if (live) {
            for (uint i = g; i < kd; i += 4) {
                const float s = decay * ssm_state[i * vd + j] + ks[i] * delta_j;
                ssm_state[i * vd + j] = s;
                o += s * qs[i];
            }
        }
        part[tid] = o;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (g == 0 && live) {
            out[((ulong)t * vh + h) * vd + j] = part[lane] + part[32 + lane] + part[64 + lane] + part[96 + lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct MulParams {
    uint total;
    uint a_len;
    uint b_len;
    uint row_broadcast;
    uint d;
};


struct Conv2dParams {
    uint in_channels;
    uint in_h;
    uint in_w;
    uint out_channels;
    uint kernel_size;
    uint stride;
    uint out_h;
    uint out_w;
    uint has_bias;
};

struct PackedGemvPairParams {
    PackedGemvParams p0;
    PackedGemvParams p1;
    uint tgs0;
    uint pad0;
    uint pad1;
    uint pad2;
};

kernel void gemv_q4_0_half_pair(device const block_q4_0* w0 [[buffer(0)]],
                                device const block_q4_0* w1 [[buffer(1)]],
                                device const float* x [[buffer(2)]],
                                device float* y0 [[buffer(3)]],
                                device float* y1 [[buffer(4)]],
                                constant PackedGemvPairParams& pp [[buffer(5)]],
                                device const float* bias0 [[buffer(6)]],
                                device const float* bias1 [[buffer(7)]],
                                device const float* res0 [[buffer(8)]],
                                device const float* res1 [[buffer(9)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    bool is_leg1 = (tg.x >= pp.tgs0);
    uint tg_x = is_leg1 ? (tg.x - pp.tgs0) : tg.x;
    constant PackedGemvParams& p = is_leg1 ? pp.p1 : pp.p0;
    device const block_q4_0* w = is_leg1 ? w1 : w0;
    device float* y = is_leg1 ? y1 : y0;
    device const float* bias = is_leg1 ? bias1 : bias0;
    device const float* res = is_leg1 ? res1 : res0;

    uint row0 = tg_x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 32;
    uint n_sub = n_blocks * 2;
    device const block_q4_0* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q4_0* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint s = lane; s < n_sub; s += 32) {
        uint b = s >> 1;
        uint half_off = (s & 1) * 8;
        device const float* xb = xb_base + b * 32 + half_off;

        float4 x_lo0 = *reinterpret_cast<device const float4*>(xb);
        float4 x_lo1 = *reinterpret_cast<device const float4*>(xb + 4);
        float4 x_hi0 = *reinterpret_cast<device const float4*>(xb + 16);
        float4 x_hi1 = *reinterpret_cast<device const float4*>(xb + 20);
        float sum_x = (x_lo0.x + x_lo0.y + x_lo0.z + x_lo0.w)
                    + (x_lo1.x + x_lo1.y + x_lo1.z + x_lo1.w)
                    + (x_hi0.x + x_hi0.y + x_hi0.z + x_hi0.w)
                    + (x_hi1.x + x_hi1.y + x_hi1.z + x_hi1.w);

        uint2 q0 = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(r0_blocks[b].qs + half_off)));
        float dot0 = dot(float4(as_type<uchar4>(q0.x & 0x0F0F0F0Fu)), x_lo0)
                   + dot(float4(as_type<uchar4>(q0.y & 0x0F0F0F0Fu)), x_lo1)
                   + dot(float4(as_type<uchar4>((q0.x >> 4) & 0x0F0F0F0Fu)), x_hi0)
                   + dot(float4(as_type<uchar4>((q0.y >> 4) & 0x0F0F0F0Fu)), x_hi1);
        acc0 += float(r0_blocks[b].d) * (dot0 - 8.0f * sum_x);

        if (has_row1) {
            uint2 q1 = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(r1_blocks[b].qs + half_off)));
            float dot1 = dot(float4(as_type<uchar4>(q1.x & 0x0F0F0F0Fu)), x_lo0)
                       + dot(float4(as_type<uchar4>(q1.y & 0x0F0F0F0Fu)), x_lo1)
                       + dot(float4(as_type<uchar4>((q1.x >> 4) & 0x0F0F0F0Fu)), x_hi0)
                       + dot(float4(as_type<uchar4>((q1.y >> 4) & 0x0F0F0F0Fu)), x_hi1);
            acc1 += float(r1_blocks[b].d) * (dot1 - 8.0f * sum_x);
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        if (has_row1) acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_q8_0_pair(device const block_q8_0* w0 [[buffer(0)]],
                           device const block_q8_0* w1 [[buffer(1)]],
                           device const float* x [[buffer(2)]],
                           device float* y0 [[buffer(3)]],
                           device float* y1 [[buffer(4)]],
                           constant PackedGemvPairParams& pp [[buffer(5)]],
                           device const float* bias0 [[buffer(6)]],
                           device const float* bias1 [[buffer(7)]],
                           device const float* res0 [[buffer(8)]],
                           device const float* res1 [[buffer(9)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    bool is_leg1 = (tg.x >= pp.tgs0);
    uint tg_x = is_leg1 ? (tg.x - pp.tgs0) : tg.x;
    constant PackedGemvParams& p = is_leg1 ? pp.p1 : pp.p0;
    device const block_q8_0* w = is_leg1 ? w1 : w0;
    device float* y = is_leg1 ? y1 : y0;
    device const float* bias = is_leg1 ? bias1 : bias0;
    device const float* res = is_leg1 ? res1 : res0;

    uint row0 = tg_x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 32;
    uint n_sub = n_blocks * 4;
    device const block_q8_0* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q8_0* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint s = lane; s < n_sub; s += 32) {
        uint b = s >> 2;
        uint j = (s & 3) * 8;
        device const float* xb = xb_base + b * 32 + j;
        float4 x0 = *reinterpret_cast<device const float4*>(xb);
        float4 x1 = *reinterpret_cast<device const float4*>(xb + 4);
        uint2 q0 = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(r0_blocks[b].qs + j)));
        acc0 += float(r0_blocks[b].d) * (dot(float4(as_type<char4>(q0.x)), x0) + dot(float4(as_type<char4>(q0.y)), x1));
        if (has_row1) {
            uint2 q1 = as_type<uint2>(ushort4(*reinterpret_cast<device const packed_ushort4*>(r1_blocks[b].qs + j)));
            acc1 += float(r1_blocks[b].d) * (dot(float4(as_type<char4>(q1.x)), x0) + dot(float4(as_type<char4>(q1.y)), x1));
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_q4_k_pair(device const block_q4_k* w0 [[buffer(0)]],
                           device const block_q4_k* w1 [[buffer(1)]],
                           device const float* x [[buffer(2)]],
                           device float* y0 [[buffer(3)]],
                           device float* y1 [[buffer(4)]],
                           constant PackedGemvPairParams& pp [[buffer(5)]],
                           device const float* bias0 [[buffer(6)]],
                           device const float* bias1 [[buffer(7)]],
                           device const float* res0 [[buffer(8)]],
                           device const float* res1 [[buffer(9)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    bool is_leg1 = (tg.x >= pp.tgs0);
    uint tg_x = is_leg1 ? (tg.x - pp.tgs0) : tg.x;
    constant PackedGemvParams& p = is_leg1 ? pp.p1 : pp.p0;
    device const block_q4_k* w = is_leg1 ? w1 : w0;
    device float* y = is_leg1 ? y1 : y0;
    device const float* bias = is_leg1 ? bias1 : bias0;
    device const float* res = is_leg1 ? res1 : res0;

    uint row0 = tg_x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    uint n_blocks = p.cols / 256;
    uint n_sub = n_blocks * 16;
    device const block_q4_k* r0_blocks = w + (ulong)row0 * n_blocks;
    device const block_q4_k* r1_blocks = has_row1 ? (w + (ulong)row1 * n_blocks) : r0_blocks;
    device const float* xb_base = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint s = lane; s < n_sub; s += 32) {
        uint b = s >> 4;
        uint g = (s >> 2) & 3;
        uint sub = s & 3;
        uint q_off = g * 32 + sub * 8;
        device const float* xb_g = xb_base + (ulong)(b * 256 + g * 64 + sub * 8);

        float4 x_lo0 = *reinterpret_cast<device const float4*>(xb_g);
        float4 x_lo1 = *reinterpret_cast<device const float4*>(xb_g + 4);
        float4 x_hi0 = *reinterpret_cast<device const float4*>(xb_g + 32);
        float4 x_hi1 = *reinterpret_cast<device const float4*>(xb_g + 36);
        float sum_x_lo = (x_lo0.x + x_lo0.y + x_lo0.z + x_lo0.w)
                       + (x_lo1.x + x_lo1.y + x_lo1.z + x_lo1.w);
        float sum_x_hi = (x_hi0.x + x_hi0.y + x_hi0.z + x_hi0.w)
                       + (x_hi1.x + x_hi1.y + x_hi1.z + x_hi1.w);

        device const block_q4_k& blk0 = r0_blocks[b];
        float2 sm0_0 = k4_scale_min(2 * g, blk0.scales);
        float2 sm1_0 = k4_scale_min(2 * g + 1, blk0.scales);
        uint2 q0 = uint2(*reinterpret_cast<device const packed_uint2*>(blk0.qs + q_off));
        float dot_lo0 = dot(float4(as_type<uchar4>(q0.x & 0x0F0F0F0Fu)), x_lo0)
                      + dot(float4(as_type<uchar4>(q0.y & 0x0F0F0F0Fu)), x_lo1);
        float dot_hi0 = dot(float4(as_type<uchar4>((q0.x >> 4) & 0x0F0F0F0Fu)), x_hi0)
                      + dot(float4(as_type<uchar4>((q0.y >> 4) & 0x0F0F0F0Fu)), x_hi1);
        acc0 += float(blk0.d) * (sm0_0.x * dot_lo0 + sm1_0.x * dot_hi0)
              - float(blk0.dmin) * (sm0_0.y * sum_x_lo + sm1_0.y * sum_x_hi);

        if (has_row1) {
            device const block_q4_k& blk1 = r1_blocks[b];
            float2 sm0_1 = k4_scale_min(2 * g, blk1.scales);
            float2 sm1_1 = k4_scale_min(2 * g + 1, blk1.scales);
            uint2 q1 = uint2(*reinterpret_cast<device const packed_uint2*>(blk1.qs + q_off));
            float dot_lo1 = dot(float4(as_type<uchar4>(q1.x & 0x0F0F0F0Fu)), x_lo0)
                          + dot(float4(as_type<uchar4>(q1.y & 0x0F0F0F0Fu)), x_lo1);
            float dot_hi1 = dot(float4(as_type<uchar4>((q1.x >> 4) & 0x0F0F0F0Fu)), x_hi0)
                          + dot(float4(as_type<uchar4>((q1.y >> 4) & 0x0F0F0F0Fu)), x_hi1);
            acc1 += float(blk1.d) * (sm0_1.x * dot_lo1 + sm1_1.x * dot_hi1)
                  - float(blk1.dmin) * (sm0_1.y * sum_x_lo + sm1_1.y * sum_x_hi);
        }
    }
    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

kernel void gemv_bf16_pair(device const ushort* w0 [[buffer(0)]],
                           device const ushort* w1 [[buffer(1)]],
                           device const float* x [[buffer(2)]],
                           device float* y0 [[buffer(3)]],
                           device float* y1 [[buffer(4)]],
                           constant PackedGemvPairParams& pp [[buffer(5)]],
                           device const float* bias0 [[buffer(6)]],
                           device const float* bias1 [[buffer(7)]],
                           device const float* res0 [[buffer(8)]],
                           device const float* res1 [[buffer(9)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    bool is_leg1 = (tg.x >= pp.tgs0);
    uint tg_x = is_leg1 ? (tg.x - pp.tgs0) : tg.x;
    constant PackedGemvParams& p = is_leg1 ? pp.p1 : pp.p0;
    device const ushort* w = is_leg1 ? w1 : w0;
    device float* y = is_leg1 ? y1 : y0;
    device const float* bias = is_leg1 ? bias1 : bias0;
    device const float* res = is_leg1 ? res1 : res0;

    uint row0 = tg_x * 16 + sg * 2;
    uint row1 = row0 + 1;
    uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    bool has_row1 = (row1 < p.rows);

    device const ushort* r0_w = w + (ulong)row0 * p.cols;
    device const ushort* r1_w = has_row1 ? (w + (ulong)row1 * p.cols) : r0_w;
    device const float* xb = x + (ulong)batch * p.cols;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    uint col_rem = p.cols % 8;
    uint col_aligned = p.cols - col_rem;

    for (uint i = lane * 8; i < col_aligned; i += 32 * 8) {
        ushort4 w0_0 = *reinterpret_cast<device const ushort4*>(r0_w + i);
        ushort4 w0_1 = *reinterpret_cast<device const ushort4*>(r0_w + i + 4);
        ushort4 w1_0 = *reinterpret_cast<device const ushort4*>(r1_w + i);
        ushort4 w1_1 = *reinterpret_cast<device const ushort4*>(r1_w + i + 4);

        float4 xb0 = *reinterpret_cast<device const float4*>(xb + i);
        float4 xb1 = *reinterpret_cast<device const float4*>(xb + i + 4);

        float4 val0_0 = as_type<float4>(uint4(w0_0) << 16);
        float4 val0_1 = as_type<float4>(uint4(w0_1) << 16);
        acc0 += dot(val0_0, xb0) + dot(val0_1, xb1);

        if (has_row1) {
            float4 val1_0 = as_type<float4>(uint4(w1_0) << 16);
            float4 val1_1 = as_type<float4>(uint4(w1_1) << 16);
            acc1 += dot(val1_0, xb0) + dot(val1_1, xb1);
        }
    }
    for (uint i = col_aligned + lane; i < p.cols; i += 32) {
        uint u0 = uint(r0_w[i]) << 16;
        acc0 += as_type<float>(u0) * xb[i];
        if (has_row1) {
            uint u1 = uint(r1_w[i]) << 16;
            acc1 += as_type<float>(u1) * xb[i];
        }
    }

    acc0 = simd_sum(acc0);
    acc1 = simd_sum(acc1);

    if (lane == 0) {
        if (p.has_bias != 0) {
            acc0 += bias[row0];
            if (has_row1) acc1 += bias[row1];
        }
        acc0 = apply_gemv_epi(acc0, (ulong)batch * p.rows + row0, p, res);
        if (has_row1) acc1 = apply_gemv_epi(acc1, (ulong)batch * p.rows + row1, p, res);
        y[(ulong)batch * p.rows + row0] = acc0;
        if (has_row1) {
            y[(ulong)batch * p.rows + row1] = acc1;
        }
    }
}

struct SampleGpuParams {
    uint vocab;
    uint top_k;
    float inv_temp;
    float top_p;
    float rep_penalty;
    uint n_recent;
    ulong rng_seed;
};

/// Logits each `sample_topk_part` threadgroup covers (256 threads x 16 values).
constant uint SAMPLE_SPAN = 4096;

/// Sampling order: higher value first, then lower id.
static inline bool sample_before(float av, uint ai, float bv, uint bi) {
    return av > bv || (av == bv && ai < bi);
}

/// The threadgroup's first `(v, i)` in sampling order (256 threads, 8 simdgroups), in every thread.
/// `sv`/`si` hold 8 entries; callers alternate two sets per round so one barrier per round suffices.
static inline void sample_tg_best(float v, uint i, threadgroup float* sv, threadgroup uint* si,
                                  uint sg, uint lane, thread float& bv, thread uint& bi) {
    const float m = simd_max(v);
    const uint mi = simd_min(v == m ? i : 0xFFFFFFFFu);
    if (lane == 0) {
        sv[sg] = m;
        si[sg] = mi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bv = sv[0];
    bi = si[0];
    for (uint s = 1; s < 8; ++s) {
        if (sample_before(sv[s], si[s], bv, bi)) {
            bv = sv[s];
            bi = si[s];
        }
    }
}

/// Exact top-`k` (`k = clamp(top_k, 1, 64)`) of each 4096-wide logit span after the repetition
/// penalty (NaN as -inf), in sampling order: `cand[part * k + j] = (value bits, id)`.
/// Grid `ceil(vocab / 4096)` threadgroups of 256 threads.
kernel void sample_topk_part(device const float* logits [[buffer(0)]],
                             device uint2* cand [[buffer(1)]],
                             device const uint* recent_ids [[buffer(2)]],
                             constant SampleGpuParams& p [[buffer(3)]],
                             uint part [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint sh_recent[64];
    threadgroup float sv[2][8];
    threadgroup uint si[2][8];
    if (tid < 64) {
        sh_recent[tid] = (tid < p.n_recent) ? recent_ids[tid] : 0xFFFFFFFFu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool has_rep = (p.rep_penalty > 0.0f && p.rep_penalty != 1.0f && p.n_recent > 0);
    const uint base = part * SAMPLE_SPAN + tid;
    float v[16];
#pragma unroll
    for (uint j = 0; j < 16; ++j) {
        const uint i = base + 256 * j;
        float x = -INFINITY;
        if (i < p.vocab) {
            x = logits[i];
            if (isnan(x)) x = -INFINITY;
            if (has_rep) {
                for (uint r = 0; r < p.n_recent && r < 64; ++r) {
                    if (sh_recent[r] == i) {
                        x = (x > 0.0f) ? (x / p.rep_penalty) : (x * p.rep_penalty);
                        break;
                    }
                }
            }
        }
        v[j] = x;
    }
    const uint k = min(max(p.top_k, 1u), 64u);
    // This thread's best value after the last one it gave up (`first`: none given up yet).
    float lv = 0.0f;
    uint li = 0;
    bool first = true, rescan = true;
    float bv = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    for (uint r = 0; r < k; ++r) {
        if (rescan) {
            bv = -INFINITY;
            bi = 0xFFFFFFFFu;
#pragma unroll
            for (uint j = 0; j < 16; ++j) {
                const uint i = base + 256 * j;
                if ((first || sample_before(lv, li, v[j], i)) && sample_before(v[j], i, bv, bi)) {
                    bv = v[j];
                    bi = i;
                }
            }
            rescan = false;
        }
        float wv;
        uint wi;
        sample_tg_best(bv, bi, sv[r & 1], si[r & 1], sg, lane, wv, wi);
        if (tid == 0) {
            cand[part * k + r] = uint2(as_type<uint>(wv), wi);
        }
        if (bi == wi) {
            lv = wv;
            li = wi;
            first = false;
            rescan = true;
        }
    }
}

/// Exact top-`k` of the `ceil(vocab / 4096) * k` part candidates, then the draw: softmax at
/// `inv_temp` over the top `k`, the top-`p` nucleus, one splitmix64 uniform from `rng_seed`.
/// Writes the token to `out_token[0]` and the `recent_ids` slot, and the kept `(id, prob bits)`
/// list to `cand[0..k]` (`(0xFFFFFFFF, 0)` past the nucleus). One threadgroup of 256 threads.
kernel void sample_topk_final(device uint2* cand [[buffer(0)]],
                              device uint* out_token [[buffer(1)]],
                              device uint* recent_ids [[buffer(2)]],
                              constant SampleGpuParams& p [[buffer(3)]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[2][8];
    threadgroup uint si[2][8];
    const uint k = min(max(p.top_k, 1u), 64u);
    const uint n = ((p.vocab + SAMPLE_SPAN - 1) / SAMPLE_SPAN) * k;
    float top_v[64];
    uint top_i[64];
    float lv = 0.0f;
    uint li = 0;
    bool first = true, rescan = true;
    float bv = -INFINITY;
    uint bi = 0xFFFFFFFFu;
    for (uint r = 0; r < k; ++r) {
        if (rescan) {
            bv = -INFINITY;
            bi = 0xFFFFFFFFu;
            for (uint c = tid; c < n; c += 256) {
                const uint2 e = cand[c];
                const float x = as_type<float>(e.x);
                if ((first || sample_before(lv, li, x, e.y)) && sample_before(x, e.y, bv, bi)) {
                    bv = x;
                    bi = e.y;
                }
            }
            rescan = false;
        }
        float wv;
        uint wi;
        sample_tg_best(bv, bi, sv[r & 1], si[r & 1], sg, lane, wv, wi);
        if (tid == 0) {
            top_v[r] = wv;
            top_i[r] = wi;
        }
        if (bi == wi) {
            lv = wv;
            li = wi;
            first = false;
            rescan = true;
        }
    }
    if (tid != 0) return;
    const float max_v = top_v[0];
    float total = 0.0f;
    for (uint j = 0; j < k; ++j) {
        float w = exp((top_v[j] - max_v) * p.inv_temp);
        top_v[j] = w;
        total += w;
    }
    const float cutoff = total * clamp(p.top_p, 1e-6f, 1.0f);
    float kept_sum = 0.0f;
    uint kept_k = 0;
    for (uint j = 0; j < k; ++j) {
        kept_sum += top_v[j];
        kept_k = j + 1;
        if (kept_sum >= cutoff) break;
    }
    ulong z = p.rng_seed + 0x9e3779b97f4a7c15UL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    z = z ^ (z >> 31);
    const float u = (float(z >> 40) * (1.0f / 16777216.0f)) * kept_sum;
    float acc = 0.0f;
    uint chosen = top_i[0];
    for (uint j = 0; j < kept_k; ++j) {
        acc += top_v[j];
        if (u <= acc) {
            chosen = top_i[j];
            break;
        }
    }
    out_token[0] = chosen;
    if (p.n_recent > 0) {
        recent_ids[uint(p.rng_seed) % min(p.n_recent, 64u)] = chosen;
    }
    for (uint j = 0; j < k; ++j) {
        cand[j] = j < kept_k ? uint2(top_i[j], as_type<uint>(top_v[j] / total)) : uint2(0xFFFFFFFFu, 0u);
    }
}

