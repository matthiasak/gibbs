use crate::graph::codec::{planar, tile, wide, Codec, Tile};

#[derive(Debug, Clone, Copy)]
pub struct GemvParams {
    pub rows_per_sg: usize,
    pub k_unroll: usize,
    /// Simdgroups per threadgroup of [`gemv_packed`]: 8 (256 threads), or 2 (64 threads, ggml-metal's
    /// K-quant launch shape).
    pub simdgroups: usize,
}

impl Default for GemvParams {
    fn default() -> Self {
        Self {
            rows_per_sg: 2,
            k_unroll: 1,
            simdgroups: 8,
        }
    }
}

impl crate::hardware::QuantFormat {
    /// Alias for the GGUF Q4_K block format.
    pub const GGUF_Q4_K: Self = Self::GGUF_Q4_K_M;

    /// Every `QuantFormat` variant.
    pub const ALL: &'static [Self] = &[
        Self::FP32,
        Self::FP16,
        Self::BF16,
        Self::FP8_E4M3,
        Self::FP8_E5M2,
        Self::NVFP4,
        Self::NVFP4_E2M1,
        Self::MXFP4,
        Self::MXFP6,
        Self::AWQ_W4A16,
        Self::GPTQ_W4A16,
        Self::GGUF_Q4_0,
        Self::GGUF_Q4_1,
        Self::GGUF_Q4_K_M,
        Self::GGUF_Q5_0,
        Self::GGUF_Q5_1,
        Self::GGUF_Q5_K_M,
        Self::GGUF_Q6_K,
        Self::GGUF_Q8_0,
        Self::GGUF_IQ4_XS,
        Self::GGUF_IQ4_NL,
        Self::GGUF_Q2_K,
        Self::GGUF_Q3_K,
        Self::GGUF_IQ3_XXS,
        Self::GGUF_IQ3_S,
        Self::GGUF_IQ2_XXS,
        Self::GGUF_IQ2_XS,
        Self::GGUF_IQ2_S,
        Self::GGUF_IQ2_M,
        Self::GGUF_IQ1_S,
        Self::GGUF_IQ1_M,
        Self::GGUF_TQ1_0,
        Self::GGUF_TQ2_0,
        Self::BitNet_B1_58,
        Self::KIVI_2Bit,
        Self::TurboQuant,
        Self::BNB_NF4,
        Self::BNB_FP4,
        Self::BNB_NF4_B64,
        Self::BNB_NF4_B128,
        Self::BNB_NF4_B256,
        Self::BNB_FP4_B64,
        Self::BNB_FP4_B128,
        Self::BNB_FP4_B256,
        Self::FP8_E4M3_B32,
        Self::MLX_AFFINE4_G32,
        Self::MLX_AFFINE4_G64,
        Self::MLX_AFFINE4_G128,
        Self::INT4_G32,
        Self::INT4_G64,
        Self::INT4_G128,
        Self::GGUF_NVFP4,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::codec::codec;
    use crate::hardware::QuantFormat;

    #[test]
    fn gemv_packed_64_thread_variant_covers_two_simdgroups() {
        let c = codec(QuantFormat::GGUF_Q4_K_M).unwrap();
        let p = GemvParams {
            simdgroups: 2,
            ..Default::default()
        };
        assert_eq!((gemv_rows_per_tg(p), gemv_threads(p)), (4, 64));
        assert_eq!(gemv_packed_name(c, p), "gemvgen_GGUF_Q4_K_M_r2_sg2");
        // The 256-thread default keeps its name and source (tuned shapes keep their cache keys).
        let d = GemvParams::default();
        assert_eq!(gemv_packed_name(c, d), "gemvgen_GGUF_Q4_K_M_r2");
    }
}

/// Rows one threadgroup of [`gemv_packed`] covers: `simdgroups` × `rows_per_sg`.
pub const fn gemv_rows_per_tg(p: GemvParams) -> usize {
    p.simdgroups * p.rows_per_sg
}

/// Threads per threadgroup of [`gemv_packed`]: 32 per simdgroup.
pub const fn gemv_threads(p: GemvParams) -> usize {
    32 * p.simdgroups
}

/// The MSL a GPU consumer of the device weight buffer includes: [`Codec::msl`], or the planar-layout decoder
/// for a codec stored [`planar`].
pub fn device_msl(codec: &Codec) -> &'static str {
    planar(codec.format).map_or(codec.msl, |p| p.msl)
}

/// How MSL addresses block `idx` (a `ulong` expression) of weights `w`: a declaration of `b` (what `decode8`
/// takes as `blk`) and the prefixes to put before `sub` (resp. the wide group index): empty unless the codec is
/// stored [`planar`], where `b` is the block's group and the prefixes select the block within it.
pub struct BlockRef {
    pub decl: String,
    pub sub: String,
    pub group: String,
}

pub fn block_ref(codec: &Codec, w: &str, idx: &str) -> BlockRef {
    match planar(codec.format) {
        None => BlockRef {
            decl: format!(
                "device const uchar* b = {w} + ({idx}) * {};",
                codec.block_bytes
            ),
            sub: String::new(),
            group: String::new(),
        },
        Some(p) => {
            let groups = tile(codec.format)
                .map(|t| t.groups)
                .or_else(|| wide(codec.format).map(|w| w.groups))
                .unwrap_or(0);
            BlockRef {
                decl: format!(
                    "const ulong bi = {idx}; device const uchar* b = {w} + (bi / {}) * {};",
                    p.blocks,
                    p.blocks * codec.block_bytes
                ),
                sub: format!("uint(bi % {}) * {} + ", p.blocks, codec.block_elems / 8),
                group: format!("uint(bi % {}) * {groups} + ", p.blocks),
            }
        }
    }
}

/// MSL common to the hand kernels and generated GEMVs: GELU variants, `PackedGemvParams` and
/// `apply_gemv_epi`, taken from `metal_kernels.metal` between its `GEMV-COMMON` markers.
pub fn gemv_common(kernels_msl: &str) -> &str {
    let begin = kernels_msl
        .find("// GEMV-COMMON-BEGIN")
        .expect("GEMV-COMMON-BEGIN marker in metal_kernels.metal");
    let end = kernels_msl
        .find("// GEMV-COMMON-END")
        .expect("GEMV-COMMON-END marker in metal_kernels.metal");
    &kernels_msl[begin..end]
}

/// MSL common to the hand kernels and generated GEMMs: `gelu_erf`, `GemmParams` and `epilogue<EPI>`,
/// taken from `metal_kernels.metal` between its `GEMM-COMMON` markers.
pub fn gemm_common(kernels_msl: &str) -> &str {
    let begin = kernels_msl
        .find("// GEMM-COMMON-BEGIN")
        .expect("GEMM-COMMON-BEGIN marker in metal_kernels.metal");
    let end = kernels_msl
        .find("// GEMM-COMMON-END")
        .expect("GEMM-COMMON-END marker in metal_kernels.metal");
    &kernels_msl[begin..end]
}

pub static CODEC_F16: Codec = Codec {
    format: crate::hardware::QuantFormat::FP16,
    block_elems: 8,
    block_bytes: 16,
    msl: "inline void decode8(device const uchar* blk, uint sub, thread float* out8) {\n    device const half* h = (device const half*)blk;\n    for (uint i = 0; i < 8; ++i) out8[i] = float(h[i]);\n}\n",
};

pub static CODEC_BF16: Codec = Codec {
    format: crate::hardware::QuantFormat::BF16,
    block_elems: 8,
    block_bytes: 16,
    msl: "inline void decode8(device const uchar* blk, uint sub, thread float* out8) {\n    device const ushort* h = (device const ushort*)blk;\n    for (uint i = 0; i < 8; ++i) out8[i] = as_type<float>(uint(h[i]) << 16);\n}\n",
};

/// Codec for [`gemm_packed`], including the trivial 8-element `decode8` for F16 and BF16 weights.
pub fn gemm_codec(f: crate::hardware::QuantFormat) -> Option<&'static Codec> {
    match f {
        crate::hardware::QuantFormat::FP16 => Some(&CODEC_F16),
        crate::hardware::QuantFormat::BF16 => Some(&CODEC_BF16),
        other => crate::graph::codec::codec(other),
    }
}

/// Name of the kernel [`gemm_packed`] emits for `codec` with `epi` (and `indexed` for MoE banks).
pub fn gemm_packed_name(codec: &Codec, epi: u32, indexed: bool) -> String {
    if indexed {
        let suffix = match epi {
            0 => "none",
            1 => "gelu",
            2 => "residual",
            3 => "silu",
            _ => "none",
        };
        format!("gemmgen_id_{:?}_{suffix}", codec.format)
    } else {
        format!("gemmgen_{:?}_e{epi}", codec.format)
    }
}

/// In-tile dequantization GEMM on the GPU's matrix units (`matmul2d` over a 128×64 output tile,
/// K step 32, 4 KiB threadgroup weight tile, 128 threads / 4 simdgroups).
/// When `indexed` is true, runs grouped expert GEMM (`grid = (ceil(slots/128), ceil(n/64), experts)`)
/// using `counts[experts]` and `slot_list[experts][slots]` produced by `moe_map_rows`.
/// Reference: ggml-metal tensor-API `kernel_mul_mm` / `kernel_mul_mm_id`.
pub fn gemm_packed(codec: &Codec, epi: u32, indexed: bool) -> String {
    let name = gemm_packed_name(codec, epi, indexed);
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = device_msl(codec).replace("decode8", &dec);
    let (sa_ty, relaxed) = match codec.format {
        crate::hardware::QuantFormat::FP32 => ("float", false),
        crate::hardware::QuantFormat::BF16 => ("bfloat", false),
        crate::hardware::QuantFormat::FP16 => ("half", false),
        _ => ("half", true),
    };
    let (be, subs) = (codec.block_elems, codec.block_elems / 8);
    if indexed {
        let (sa_ty, relaxed) = ("float", false);
        let br = block_ref(
            codec,
            "w",
            &format!("((ulong)e * p.rows + col) * (p.cols / {be}) + s / {subs}"),
        );
        let (decl, sub) = (&br.decl, &br.sub);
        let epi_expr = match epi {
            1 => "gelu_erf(val)",
            3 => "val / (1.0f + exp(-val))",
            _ => "val",
        };
        return format!(
            r#"
#if defined(GIBBS_MPP)
using namespace mpp::tensor_ops;
{codec_msl}

struct MoeGemmParams {{
    uint rows;
    uint cols;
    uint slots;
    uint top_k;
    uint experts;
    uint x_per_slot;
}};

kernel void {name}(
    device const float* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const uint* counts [[buffer(2)]],
    device const uint* slot_list [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant MoeGemmParams& p [[buffer(5)]],
    threadgroup uchar* shmem [[threadgroup(0)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    const uint e = tg.z;
    if (e >= p.experts) return;
    const uint cnt = counts[e];
    const uint m0 = tg.x * 128;
    const uint n0 = tg.y * 64;
    if (m0 >= cnt || n0 >= p.rows) return;
    const uint tile_m = min(128u, cnt - m0);
    threadgroup {sa_ty}* sa = (threadgroup {sa_ty}*)shmem;
    threadgroup float* sb = (threadgroup float*)(shmem + 64 * 32 * sizeof({sa_ty}));
    device const uint* elist = slot_list + (ulong)e * p.slots + m0;

    auto tA = tensor(sa, dextents<int32_t, 2>(32, 64), array<int, 2>({{1, 32}}));
    auto tB = tensor(sb, dextents<int32_t, 2>(32, 128), array<int, 2>({{1, 32}}));
    constexpr auto desc = matmul2d_descriptor(
        128, 64, static_cast<int>(dynamic_extent), false, true, {relaxed},
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto c = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {{
        if (c.is_valid_element(i)) c[i] = 0.0f;
    }}

    for (uint k0 = 0; k0 < p.cols; k0 += 32) {{
        for (uint work = tid; work < 256; work += 128) {{
            const uint feat = work / 4;
            const uint chunk = work % 4;
            const uint col = n0 + feat;
            const uint gk = k0 + chunk * 8;
            threadgroup {sa_ty}* dst = sa + feat * 32 + chunk * 8;
            if (col < p.rows && gk + 8 <= p.cols) {{
                const uint s = gk / 8;
                {decl}
                float v[8];
                {dec}(b, {sub}s % {subs}, v);
                *((threadgroup float4*)dst) = float4(v[0], v[1], v[2], v[3]);
                *((threadgroup float4*)(dst + 4)) = float4(v[4], v[5], v[6], v[7]);
            }} else if (col < p.rows && gk < p.cols) {{
                const uint s = gk / 8;
                {decl}
                float v[8];
                {dec}(b, {sub}s % {subs}, v);
                #pragma unroll
                for (uint i = 0; i < 8; ++i) {{
                    dst[i] = (gk + i < p.cols) ? {sa_ty}(v[i]) : {sa_ty}(0);
                }}
            }} else {{
                #pragma unroll
                for (uint i = 0; i < 8; ++i) dst[i] = {sa_ty}(0);
            }}
        }}
        for (uint work = tid; work < 512; work += 128) {{
            const uint r = work / 4;
            const uint chunk = work % 4;
            const uint gk = k0 + chunk * 8;
            threadgroup float* dst = sb + r * 32 + chunk * 8;
            if (r < tile_m && gk + 8 <= p.cols) {{
                const uint slot = elist[r];
                const uint x_row = p.x_per_slot != 0 ? slot : (slot / p.top_k);
                device const float* xr = x + (ulong)x_row * p.cols + gk;
                *((threadgroup float4*)dst) = *((device const float4*)xr);
                *((threadgroup float4*)(dst + 4)) = *((device const float4*)(xr + 4));
            }} else if (r < tile_m && gk < p.cols) {{
                const uint slot = elist[r];
                const uint x_row = p.x_per_slot != 0 ? slot : (slot / p.top_k);
                device const float* xr = x + (ulong)x_row * p.cols + gk;
                #pragma unroll
                for (uint i = 0; i < 8; ++i) {{
                    dst[i] = (gk + i < p.cols) ? xr[i] : 0.0f;
                }}
            }} else {{
                #pragma unroll
                for (uint i = 0; i < 8; ++i) dst[i] = 0.0f;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int kExt = int(min(32u, p.cols - k0));
        auto xv = tensor(sb, dextents<int32_t, 2>(kExt, int(tile_m)), array<int, 2>({{1, 32}}));
        auto sav = tensor(sa, dextents<int32_t, 2>(kExt, 64), array<int, 2>({{1, 32}}));
        mm.run(xv, sav, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {{
        if (c.is_valid_element(i)) {{
            auto idx = c.get_multidimensional_index(i);
            const uint col = n0 + uint(idx[0]);
            const uint r = uint(idx[1]);
            if (r < tile_m && col < p.rows) {{
                const uint slot = elist[r];
                const float val = c[i];
                out[(ulong)slot * p.rows + col] = {epi_expr};
            }}
        }}
    }}
}}
#endif
"#
        );
    }
    let br = block_ref(
        codec,
        "w",
        &format!("(ulong)col * (p.k / {be}) + s / {subs}"),
    );
    let (decl, sub) = (&br.decl, &br.sub);
    format!(
        r#"
#if defined(GIBBS_MPP)
using namespace mpp::tensor_ops;
{codec_msl}

kernel void {name}(
    device float* x [[buffer(0)]],
    device const uchar* w [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* res [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant GemmParams& p [[buffer(5)]],
    threadgroup {sa_ty}* sa [[threadgroup(0)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    const uint m0 = tg.x * 128;
    const uint n0 = tg.y * 64;
    if (m0 >= p.m || n0 >= p.n) return;
    const int tile_m = int(min(128u, p.m - m0));
    const int tile_n = int(min(64u, p.n - n0));

    auto tA = tensor(sa, dextents<int32_t, 2>(32, 64), array<int, 2>({{1, 32}}));
    auto tB = tensor(x, dextents<int32_t, 2>(int(p.k), int(p.m)), array<int, 2>({{1, int(p.k)}}));
    constexpr auto desc = matmul2d_descriptor(
        128, 64, static_cast<int>(dynamic_extent), false, true, {relaxed},
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<4>> mm;
    auto c = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {{
        if (c.is_valid_element(i)) c[i] = 0.0f;
    }}

    for (uint k0 = 0; k0 < p.k; k0 += 32) {{
        for (uint work = tid; work < 256; work += 128) {{
            const uint feat = work / 4;
            const uint chunk = work % 4;
            const uint col = n0 + feat;
            const uint gk = k0 + chunk * 8;
            threadgroup {sa_ty}* dst = sa + feat * 32 + chunk * 8;
            if (col < p.n && gk < p.k) {{
                const uint s = gk / 8;
                {decl}
                float v[8];
                {dec}(b, {sub}s % {subs}, v);
                #pragma unroll
                for (uint i = 0; i < 8; ++i) {{
                    dst[i] = (gk + i < p.k) ? {sa_ty}(v[i]) : {sa_ty}(0);
                }}
            }} else {{
                #pragma unroll
                for (uint i = 0; i < 8; ++i) dst[i] = {sa_ty}(0);
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int kExt = int(min(32u, p.k - k0));
        auto xv = tensor(x + (ulong)m0 * p.k + k0, dextents<int32_t, 2>(kExt, tile_m), array<int, 2>({{1, int(p.k)}}));
        auto sav = tensor(sa, dextents<int32_t, 2>(kExt, 64), array<int, 2>({{1, 32}}));
        mm.run(xv, sav, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    #pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i) {{
        if (c.is_valid_element(i)) {{
            auto idx = c.get_multidimensional_index(i);
            uint col = n0 + uint(idx[0]), row = m0 + uint(idx[1]);
            if (row < p.m && col < p.n) c[i] = epilogue<{epi}>(c[i], row, col, p, bias, res);
        }}
    }}
    auto mo = tensor(out + (ulong)m0 * p.n + n0, dextents<int32_t, 2>(tile_n, tile_m), array<int, 2>({{1, int(p.n)}}));
    c.store(mo);
}}
#endif
"#
    )
}

/// Name of the kernel [`gemv_packed`] emits for `codec` with `p` (`_sg2` suffix off the 8-simdgroup default).
pub fn gemv_packed_name(codec: &Codec, p: GemvParams) -> String {
    let sg = if p.simdgroups == 8 {
        String::new()
    } else {
        format!("_sg{}", p.simdgroups)
    };
    format!("gemvgen_{:?}_r{}{sg}", codec.format, p.rows_per_sg)
}

/// Name of the kernel [`dequant_packed`] emits for `codec`.
pub fn dequant_packed_name(codec: &Codec) -> String {
    format!("dequantgen_{:?}", codec.format)
}

/// Expands a packed `[rows, cols]` weight to f32: buffers `w, out`, bytes `(rows, cols)`, one thread
/// per 8-element sub-block (`ceil(rows * cols / 8)` threads). Prefill GEMMs read the result.
pub fn dequant_packed(codec: &Codec) -> String {
    let name = dequant_packed_name(codec);
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = device_msl(codec).replace("decode8", &dec);
    let (be, subs) = (codec.block_elems, codec.block_elems / 8);
    let br = block_ref(
        codec,
        "w",
        &format!("(ulong)row * (dims.y / {be}) + s / {subs}"),
    );
    let (decl, sub) = (&br.decl, &br.sub);
    format!(
        r#"
{codec_msl}

kernel void {name}(
    device const uchar* w [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint2& dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    const uint per_row = dims.y / 8;
    if (gid >= dims.x * per_row) return;
    const uint row = gid / per_row, s = gid % per_row;
    {decl}
    float v[8];
    {dec}(b, {sub}s % {subs}, v);
    device float* o = out + (ulong)row * dims.y + s * 8;
    for (uint i = 0; i < 8; ++i) o[i] = v[i];
}}
"#
    )
}

/// A packed-weight GEMV with the `Step::GemvPacked` contract: buffers `w, x, y, PackedGemvParams,
/// bias, res`; grid `(ceil(rows / gemv_rows_per_tg), batch)` of `gemv_threads` threads. Lane `l` of a
/// simdgroup walks each row's 8-element sub-blocks `l, l+32, …`, decodes them with the codec's
/// `decode8`, and the simdgroup sums the row. Bias and the fused epilogue are applied by lane 0.
pub fn gemv_packed(codec: &Codec, p: GemvParams) -> String {
    let name = gemv_packed_name(codec, p);
    // Every codec defines `decode8`; one library holds several codecs, so each gets its own name.
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = device_msl(codec).replace("decode8", &dec);
    let r = p.rows_per_sg;
    let be = codec.block_elems;
    let loop_src = gemv_loop(codec, p, &dec, "(ulong)(row0", true);
    let store_row = |acc: &str, row: &str, ind: &str| {
        format!(
            "{ind}float a = {acc};\n{ind}if (p.has_bias != 0) a += bias[{row}];\n{ind}ulong o = (ulong)batch * p.rows + {row};\n{ind}y[o] = apply_gemv_epi(a, o, p, res);\n"
        )
    };
    // A tile codec splits the simdgroup's lanes across its `r` rows (`32 / r` lanes per row, summed by
    // xor-shuffles): its row has few 16-byte tiles, so letting every lane serve one row keeps all of them busy.
    let (acc_decl, sums, guard, store) = if tile(codec.format).is_some() {
        let lpr = 32 / r;
        let shuffles: String = std::iter::successors(Some(lpr / 2), |o| Some(o / 2))
            .take_while(|&o| o > 0)
            .map(|o| format!("    acc0 += simd_shuffle_xor(acc0, {o});\n"))
            .collect();
        (
            format!("const uint sl = lane % {lpr}, rl = lane / {lpr}, myrow = row0 + rl;\n    float acc0 = 0.0f;"),
            shuffles,
            format!("sl == 0 && myrow < p.rows"),
            store_row("acc0", "myrow", "        "),
        )
    } else {
        let store: String = (0..r)
            .map(|i| {
                format!(
                    "        if (row0 + {i} < p.rows) {{\n{}        }}\n",
                    store_row(
                        &format!("simd_sum_{i}"),
                        &format!("row0 + {i}"),
                        "            "
                    )
                )
            })
            .collect();
        (
            (0..r)
                .map(|i| format!("float acc{i} = 0.0f;"))
                .collect::<Vec<_>>()
                .join(" "),
            (0..r)
                .map(|i| format!("    float simd_sum_{i} = simd_sum(acc{i});\n"))
                .collect(),
            "lane == 0".to_string(),
            store,
        )
    };
    format!(
        r#"
{codec_msl}
{wide_msl}
kernel void {name}(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant PackedGemvParams& p [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device const float* res [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    const uint row0 = tg.x * {rows_per_tg} + sg * {r};
    const uint batch = tg.y;
    if (row0 >= p.rows || batch >= p.batch_size) return;
    const uint blocks_per_row = p.cols / {be};
    device const float* xb = x + (ulong)batch * p.cols;
    {acc_decl}
{loop_src}{sums}    if ({guard}) {{
{store}    }}
}}
"#,
        rows_per_tg = gemv_rows_per_tg(p),
        wide_msl = wide_msl(codec, &dec, true),
    )
}

fn wide_msl(codec: &Codec, dec: &str, allow_tile: bool) -> String {
    wide(codec.format)
        .map(|w| {
            w.msl
                .replace("decode_group", &dec.replace("decode8", "decode_group"))
                .replace("dot_group", &dec.replace("decode8", "dot_group"))
                .replace("group_seg", &dec.replace("decode8", "group_seg"))
        })
        .or_else(|| {
            allow_tile.then(|| tile(codec.format)).flatten().map(|t| {
                ["group_seg", "tile_load", "tile_scale", "tile_seg"]
                    .iter()
                    .fold(t.msl.to_string(), |m, n| {
                        m.replace(n, &dec.replace("decode8", n))
                    })
            })
        })
        .unwrap_or_default()
}

/// The lane loop of [`gemv_packed`] / [`gemv_indexed`]: one 8-element sub-block per iteration (`decode8`), or,
/// for a codec with a [`wide`] decode, one group of `segs` sub-blocks (its bytes are read once, not once per sub-block).
fn gemv_loop(codec: &Codec, p: GemvParams, dec: &str, row_base: &str, allow_tile: bool) -> String {
    let (be, r) = (codec.block_elems, p.rows_per_sg);
    // `decode` gets the block's prefixes (`BlockRef::sub` for decode8, `group` for decode_group).
    let row = |i: usize, decode: &dyn Fn(&BlockRef) -> String, dots: &str, v_len: u32| {
        let br = block_ref(
            codec,
            "w",
            &format!("{row_base} + {i}) * blocks_per_row + blk"),
        );
        format!(
            "        if (row0 + {i} < p.rows) {{\n            {}\n            float v[{v_len}];\n            {}\n{dots}        }}\n",
            br.decl,
            decode(&br)
        )
    };
    if allow_tile {
        if let Some(t) = tile(codec.format) {
            return tile_loop(codec, t, p, dec);
        }
    }
    let Some(wd) = wide(codec.format) else {
        let rows: String = (0..r)
            .map(|i| {
                row(
                    i,
                    &|br| format!("{dec}(b, {}sub, v);", br.sub),
                    &format!("            acc{i} += dot(float4(v[0], v[1], v[2], v[3]), x0) + dot(float4(v[4], v[5], v[6], v[7]), x1);\n"),
                    8,
                )
            })
            .collect();
        return format!(
            "    const uint subs = p.cols / 8;\n    for (uint s = lane; s < subs; s += 32) {{\n        float4 x0 = *(device const float4*)(xb + s * 8);\n        float4 x1 = *(device const float4*)(xb + s * 8 + 4);\n        const uint blk = (s * 8) / {be};\n        const uint sub = ((s * 8) % {be}) / 8;\n{rows}    }}\n"
        );
    };
    let (gdec, gseg) = (
        dec.replace("decode8", "decode_group"),
        dec.replace("decode8", "group_seg"),
    );
    let xs: String = (0..wd.segs)
        .map(|k| {
            format!(
                "        {{ const uint e = blk * {be} + {gseg}(g, {k}); xs[{}] = *(device const float4*)(xb + e); xs[{}] = *(device const float4*)(xb + e + 4); }}\n",
                2 * k,
                2 * k + 1
            )
        })
        .collect();
    let groups = wd.groups;
    if wd.fused_dot {
        let gdot = dec.replace("decode8", "dot_group");
        let rows: String = (0..r)
            .map(|i| {
                let br = block_ref(
                    codec,
                    "w",
                    &format!("{row_base} + {i}) * blocks_per_row + blk"),
                );
                format!(
                    "        if (row0 + {i} < p.rows) {{\n            {}\n            acc{i} += {gdot}(b, {}g, xs);\n        }}\n",
                    br.decl, br.group
                )
            })
            .collect();
        return format!(
            "    const uint groups = blocks_per_row * {groups};\n    for (uint s = lane; s < groups; s += 32) {{\n        const uint blk = s / {groups};\n        const uint g = s % {groups};\n        float4 xs[{}];\n{xs}{rows}    }}\n",
            2 * wd.segs
        );
    }
    let nseg = wd.segs as usize;
    let rows: String = (0..r)
        .map(|i| {
            let seg_dot = |k: usize| {
                let o = 8 * k;
                format!(
                    "dot(float4(v[{o}], v[{}], v[{}], v[{}]), xs[{}]) + dot(float4(v[{}], v[{}], v[{}], v[{}]), xs[{}])",
                    o + 1, o + 2, o + 3, 2 * k, o + 4, o + 5, o + 6, o + 7, 2 * k + 1
                )
            };
            let mut dots = String::new();
            if wd.uniform {
                let sum = (0..nseg).map(|k| format!("({})", seg_dot(k))).collect::<Vec<_>>().join(" + ");
                dots += &format!("            acc{i} += v[{}] * ({sum});\n", 8 * nseg);
            } else {
                for k in 0..nseg {
                    dots += &format!(
                        "            acc{i} += v[{}] * ({}) + v[{}] * sx{k};\n",
                        8 * nseg + k, seg_dot(k), 9 * nseg + k
                    );
                }
            }
            let v_len = if wd.uniform { 8 * wd.segs + 1 } else { 10 * wd.segs };
            row(i, &|br| format!("{gdec}(b, {}g, v);", br.group), &dots, v_len)
        })
        .collect();
    let sx: String = if wd.uniform || wd.fused_dot {
        String::new()
    } else {
        (0..nseg)
            .map(|k| {
                format!(
                    "        const float sx{k} = dot(xs[{}], float4(1.0f)) + dot(xs[{}], float4(1.0f));\n",
                    2 * k,
                    2 * k + 1
                )
            })
            .collect()
    };
    format!(
        "    const uint groups = blocks_per_row * {groups};\n    for (uint s = lane; s < groups; s += 32) {{\n        const uint blk = s / {groups};\n        const uint g = s % {groups};\n        float4 xs[{}];\n{xs}{sx}{rows}    }}\n",
        2 * wd.segs
    )
}

/// The lane loop of [`gemv_packed`] for a [`Tile`] codec: lane `l` serves row `myrow` (`32 / rows_per_sg`
/// lanes per row), walks that row's tiles `sl, sl + lpr, …`, loads the tile and its scale, then decodes one
/// 8-element segment at a time against `x` and accumulates `scale * (dot(codes, x) - sum(x))`.
fn tile_loop(codec: &Codec, t: &Tile, p: GemvParams, dec: &str) -> String {
    let (be, r, segs) = (codec.block_elems, p.rows_per_sg, t.segs);
    let lpr = 32 / r;
    assert!(
        segs as usize % r == 0,
        "{} rows per simdgroup do not divide {segs} segments",
        r
    );
    let name = |n: &str| dec.replace("decode8", n);
    let (seg, load, scale, tseg) = (
        name("group_seg"),
        name("tile_load"),
        name("tile_scale"),
        name("tile_seg"),
    );
    let br = block_ref(codec, "w", "(ulong)myrow * blocks_per_row + blk");
    let body: String = (0..segs)
        .map(|k| {
            format!(
                "        {{\n            const uint e = blk * {be} + {seg}(g, {k});\n            const float4 xa = *(device const float4*)(xb + e), xc = *(device const float4*)(xb + e + 4);\n            float v[8];\n            {tseg}(q, {k}, v);\n            t += dot(float4(v[0], v[1], v[2], v[3]), xa) + dot(float4(v[4], v[5], v[6], v[7]), xc);\n        }}\n"
            )
        })
        .collect();
    let groups = t.groups;
    if t.b == 0.0 {
        return format!(
            "    const uint tiles = blocks_per_row * {groups};\n    for (uint s = sl; s < tiles; s += {lpr}) {{\n        const uint blk = s / {groups};\n        const uint g = s % {groups};\n        uint4 q = uint4(0);\n        float d = 0.0f, t = 0.0f;\n        if (myrow < p.rows) {{\n            {}\n            q = {load}(b, {}g);\n            d = {scale}(b, {}g);\n        }}\n{body}        acc0 += d * ({a:?}f * t);\n    }}\n",
            br.decl, br.group, br.group, a = t.a
        );
    }
    // The lanes serving the same tile of different rows (`lane % lpr` equal) split sum(x) over the tile
    // between them (`segs / r` segments each) and add the parts with xor-shuffles.
    let spr = segs as usize / r;
    let sx_parts: String = (0..spr)
        .map(|m| {
            format!(
                "        {{\n            const uint e = blk * {be} + {seg}(g, rl * {spr} + {m});\n            sx += *(device const float4*)(xb + e) + *(device const float4*)(xb + e + 4);\n        }}\n"
            )
        })
        .collect();
    let sx_sum: String = std::iter::successors(Some(lpr), |o| Some(o * 2))
        .take_while(|&o| o < 32)
        .map(|o| format!("        sxs += simd_shuffle_xor(sxs, {o});\n"))
        .collect();
    format!(
        "    const uint tiles = blocks_per_row * {groups};\n    for (uint s = sl; s < tiles; s += {lpr}) {{\n        const uint blk = s / {groups};\n        const uint g = s % {groups};\n        uint4 q = uint4(0);\n        float d = 0.0f, t = 0.0f;\n        if (myrow < p.rows) {{\n            {}\n            q = {load}(b, {}g);\n            d = {scale}(b, {}g);\n        }}\n        float4 sx = float4(0.0f);\n{sx_parts}        float sxs = dot(sx, float4(1.0f));\n{sx_sum}{body}        acc0 += d * ({a:?}f * t - {b:?}f * sxs);\n    }}\n",
        br.decl, br.group, br.group, a = t.a, b = t.b
    )
}

/// Name of the kernel [`gemv_indexed`] emits for `codec` with `p`.
pub fn gemv_indexed_name(codec: &Codec, p: GemvParams) -> String {
    format!("gemvgen_{:?}_r{}_ix", codec.format, p.rows_per_sg)
}

/// [`gemv_packed`] over an expert bank: the weight is `experts` stacked `[rows, cols]` matrices and
/// slot `s` (grid y) multiplies row `s` of `x` (`x_per_slot != 0`) or row `s / top_k` by the matrix
/// `ids[s]`, writing row `s` of `y` (`[slots, rows]`). Buffers `w, x, y, MoeGemvParams {rows, cols,
/// slots, top_k, experts, x_per_slot}, ids`; grid `(ceil(rows / gemv_rows_per_tg), slots)` of 256
/// threads. A slot whose id is not below `experts` writes zeros.
pub fn gemv_indexed(codec: &Codec, p: GemvParams) -> String {
    let name = gemv_indexed_name(codec, p);
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = device_msl(codec).replace("decode8", &dec);
    let r = p.rows_per_sg;
    let be = codec.block_elems;
    let acc_decl = (0..r)
        .map(|i| format!("float acc{i} = 0.0f;"))
        .collect::<Vec<_>>()
        .join(" ");
    let loop_src = gemv_loop(codec, p, &dec, "(bank_row + row0", false);
    let sums: String = (0..r)
        .map(|i| format!("    float simd_sum_{i} = simd_sum(acc{i});\n"))
        .collect();
    let store: String = (0..r)
            .map(|i| {
                format!("        if (row0 + {i} < p.rows) y[(ulong)slot * p.rows + row0 + {i}] = simd_sum_{i};\n")
            })
            .collect();
    format!(
        r#"
{codec_msl}
{wide_msl}

struct MoeGemvParams {{
    uint rows;
    uint cols;
    uint slots;
    uint top_k;
    uint experts;
    uint x_per_slot;
}};

kernel void {name}(
    device const uchar* w [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant MoeGemvParams& p [[buffer(3)]],
    device const uint* ids [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {{
    const uint row0 = tg.x * {rows_per_tg} + sg * {r};
    const uint slot = tg.y;
    if (row0 >= p.rows || slot >= p.slots) return;
    const uint expert = ids[slot];
    if (expert >= p.experts) {{
        if (lane == 0) {{
            for (uint i = 0; i < {r}; i++) if (row0 + i < p.rows) y[(ulong)slot * p.rows + row0 + i] = 0.0f;
        }}
        return;
    }}
    const ulong bank_row = (ulong)expert * p.rows;
    const uint blocks_per_row = p.cols / {be};
    device const float* xb = x + (ulong)(p.x_per_slot != 0 ? slot : slot / p.top_k) * p.cols;
    {acc_decl}
{loop_src}{sums}    if (lane == 0) {{
{store}    }}
}}
"#,
        rows_per_tg = gemv_rows_per_tg(p),
        wide_msl = wide_msl(codec, &dec, false),
    )
}

/// Name of the cooperative decode MLP megakernel for `(codec, is_geglu)`.
pub fn mlp_megakernel_name(codec: &Codec, is_geglu: bool) -> String {
    let act = if is_geglu { "geglu" } else { "swiglu" };
    format!("mlp_mega_{:?}_{act}", codec.format)
}

/// Cooperative decode MLP megakernel (`m == 1`, `hidden <= 4096`): each threadgroup normalizes
/// `x * norm_w` into `threadgroup float shared_x[4096]` and computes a tile of `SwiGLU(gate(shared_x), up(shared_x))`.
pub fn mlp_megakernel_msl(codec: &Codec, is_geglu: bool) -> String {
    let name = mlp_megakernel_name(codec, is_geglu);
    let dec = format!("decode8_{:?}", codec.format);
    let codec_msl = device_msl(codec).replace("decode8", &dec);
    let (be, subs) = (codec.block_elems, codec.block_elems / 8);
    let gate_br = block_ref(codec, "gate_w", "row * blocks_per_row + b");
    let up_br = block_ref(codec, "up_w", "row * blocks_per_row + b");
    let act_expr = if is_geglu {
        "u * (0.5f * g * (1.0f + tanh(0.7978845608f * (g + 0.044715f * g * g * g))))"
    } else {
        "u * (g / (1.0f + exp(-g)))"
    };
    format!(
        r#"
{codec_msl}

struct MlpMegaParams {{
    uint hidden;
    uint inter;
    float eps;
}};

kernel void {name}(
    device const float* x [[buffer(0)]],
    device const float* norm_w [[buffer(1)]],
    device const uchar* gate_w [[buffer(2)]],
    device const uchar* up_w [[buffer(3)]],
    device float* inter_out [[buffer(4)]],
    constant MlpMegaParams& p [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]]
) {{
    threadgroup float shared_x[4096];
    threadgroup float sq_part[8];
    float sq = 0.0f;
    for (uint i = tid; i < p.hidden; i += 256) {{
        float v = x[i];
        sq += v * v;
    }}
    sq = simd_sum(sq);
    if (lane == 0) sq_part[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_sq = sq_part[0] + sq_part[1] + sq_part[2] + sq_part[3]
                   + sq_part[4] + sq_part[5] + sq_part[6] + sq_part[7];
    float inv_rms = rsqrt(total_sq / float(p.hidden) + p.eps);
    for (uint i = tid; i < p.hidden; i += 256) {{
        shared_x[i] = x[i] * inv_rms * norm_w[i];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg * 8 + sg;
    if (row >= p.inter) return;
    const uint blocks_per_row = p.hidden / {be};
    float g = 0.0f;
    float u = 0.0f;
    for (uint s = lane; s < blocks_per_row * {subs}; s += 32) {{
        const ulong b = s / {subs};
        const uint sub = s % {subs};
        const uint col = b * {be} + sub * 8;
        float gw[8], uw[8];
        {{ {gate_decl} {dec}(b, {gate_sub}sub, gw); }}
        {{ {up_decl} {dec}(b, {up_sub}sub, uw); }}
        #pragma unroll
        for (uint i = 0; i < 8; ++i) {{
            float xv = shared_x[col + i];
            g += gw[i] * xv;
            u += uw[i] * xv;
        }}
    }}
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) {{
        inter_out[row] = {act_expr};
    }}
}}
"#,
        gate_decl = gate_br.decl,
        gate_sub = gate_br.sub,
        up_decl = up_br.decl,
        up_sub = up_br.sub,
    )
}
