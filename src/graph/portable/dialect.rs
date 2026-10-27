//! The three dialects one `.gpk` kernel source compiles as: OpenCL C 1.2, CUDA C++ (NVRTC) and HIP
//! C++ (hipRTC). Each prelude defines the macros and fixed-width types the kernels are written in.
//!
//! Kernel source conventions: `KERNEL name(..)` entry points; `GLOBAL` on device pointers; `LOCAL`
//! on workgroup arrays declared at kernel scope and `LPTR` on pointers to them; `DEVFN` on helper
//! functions; `CONSTANT` on program-scope tables; `GID/LID/GRP/LSZ/NGRP(i)` index queries;
//! `BARRIER()`; `ATOMIC_ADD_F32(p, v)`; `AS_FLOAT`/`AS_UINT` bit casts; C99 `*f` math names; types
//! `u8 i8 u16 i16 u32 i32 u64 i64`. No half arithmetic, no subgroups.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dialect {
    OpenCl,
    Cuda,
    Hip,
}

const OPENCL: &str = r#"
#define KERNEL __kernel void
#define GLOBAL __global
#define LOCAL __local
#define LPTR __local
#define CONSTANT __constant
#define DEVFN
#define RESTRICT restrict
#define GID(i) ((u32)get_global_id(i))
#define LID(i) ((u32)get_local_id(i))
#define GRP(i) ((u32)get_group_id(i))
#define LSZ(i) ((u32)get_local_size(i))
#define NGRP(i) ((u32)get_num_groups(i))
#define BARRIER() barrier(CLK_LOCAL_MEM_FENCE)
#define AS_FLOAT(u) as_float((uint)(u))
#define AS_UINT(f) as_uint((float)(f))
typedef uchar u8;
typedef char i8;
typedef ushort u16;
typedef short i16;
typedef uint u32;
typedef int i32;
typedef ulong u64;
typedef long i64;
#define expf(x) exp(x)
#define logf(x) log(x)
#define log1pf(x) log1p(x)
#define tanhf(x) tanh(x)
#define sqrtf(x) sqrt(x)
#define erff(x) erf(x)
#define floorf(x) floor(x)
#define ceilf(x) ceil(x)
#define rintf(x) rint(x)
#define fabsf(x) fabs(x)
#define fmaxf(a, b) fmax(a, b)
#define fminf(a, b) fmin(a, b)
#define cosf(x) cos(x)
#define sinf(x) sin(x)
#define powf(a, b) pow(a, b)
void ATOMIC_ADD_F32(volatile __global float* p, float v) {
    volatile __global uint* u = (volatile __global uint*)p;
    uint old = *u, prev;
    do {
        prev = old;
        old = atomic_cmpxchg(u, prev, as_uint(as_float(prev) + v));
    } while (old != prev);
}
"#;

/// Shared by CUDA and HIP: both compile C++ with `__global__` entry points and `extern "C"` names.
const CUDA_LIKE: &str = r#"
#define KERNEL extern "C" __global__ void
#define GLOBAL
#define LOCAL __shared__
#define LPTR
#define CONSTANT __device__ const
#define DEVFN __device__ __forceinline__
#define RESTRICT __restrict__
typedef unsigned char u8;
typedef signed char i8;
typedef unsigned short u16;
typedef short i16;
typedef unsigned int u32;
typedef int i32;
typedef unsigned long long u64;
typedef long long i64;
__device__ __forceinline__ u32 gpk_lid(int i) { return i == 0 ? threadIdx.x : i == 1 ? threadIdx.y : threadIdx.z; }
__device__ __forceinline__ u32 gpk_grp(int i) { return i == 0 ? blockIdx.x : i == 1 ? blockIdx.y : blockIdx.z; }
__device__ __forceinline__ u32 gpk_lsz(int i) { return i == 0 ? blockDim.x : i == 1 ? blockDim.y : blockDim.z; }
__device__ __forceinline__ u32 gpk_ngrp(int i) { return i == 0 ? gridDim.x : i == 1 ? gridDim.y : gridDim.z; }
#define LID(i) gpk_lid(i)
#define GRP(i) gpk_grp(i)
#define LSZ(i) gpk_lsz(i)
#define NGRP(i) gpk_ngrp(i)
#define GID(i) (gpk_grp(i) * gpk_lsz(i) + gpk_lid(i))
#define BARRIER() __syncthreads()
#define AS_FLOAT(u) __uint_as_float((u32)(u))
#define AS_UINT(f) __float_as_uint((float)(f))
#define ATOMIC_ADD_F32(p, v) atomicAdd((p), (v))
"#;

/// The dialect prelude: macros and types every `.gpk` source assumes.
pub fn prelude(d: Dialect) -> &'static str {
    match d {
        Dialect::OpenCl => OPENCL,
        Dialect::Cuda | Dialect::Hip => CUDA_LIKE,
    }
}
