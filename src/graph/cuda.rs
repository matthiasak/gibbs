//! NVIDIA CUDA backend: [`CudaApi`] implements the portable [`GpuApi`] over the CUDA driver API
//! (`libcuda`) and NVRTC (`libnvrtc`), both loaded at runtime through [`DynLib`] (no build-time
//! CUDA SDK). The portable kernel source compiles to PTX for the device's compute capability;
//! every launch goes to one blocking stream, which stream capture records into CUDA graphs.

use crate::graph::plan::DynLib;
use crate::graph::portable::api::{Arg, Buf, GpuApi, Kernel, Module, Replay};
use crate::graph::portable::dialect::Dialect;
use crate::graph::{portable, DType, Op};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::ptr::{null, null_mut};
use std::sync::{Arc, LazyLock};

type P = *mut c_void;
type R = c_int;

/// Most kernel arguments one launch passes.
const MAX_ARGS: usize = 64;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;

/// Resolves `name` in `lib` as a function pointer of type `F`.
///
/// # Safety
/// `F` must be the symbol's real C signature.
unsafe fn sym<F: Copy>(lib: &DynLib, name: &CStr) -> Option<F> {
    let p = lib.sym_ptr(name)?;
    assert_eq!(size_of::<F>(), size_of::<P>());
    Some(std::mem::transmute_copy(&p))
}

fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        return "unknown".into();
    }
    // SAFETY: CUDA/NVRTC error strings are static NUL-terminated strings.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// `cuGraphInstantiateWithFlags` (CUDA >= 11.4), else the CUDA 11 `cuGraphInstantiate_v2`.
#[derive(Clone, Copy)]
enum Instantiate {
    WithFlags(unsafe extern "C" fn(*mut P, P, u64) -> R),
    V2(unsafe extern "C" fn(*mut P, P, *mut P, *mut c_char, usize) -> R),
}

struct Cu {
    init: unsafe extern "C" fn(c_uint) -> R,
    device_get_count: unsafe extern "C" fn(*mut c_int) -> R,
    device_get: unsafe extern "C" fn(*mut c_int, c_int) -> R,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> R,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> R,
    primary_ctx_retain: unsafe extern "C" fn(*mut P, c_int) -> R,
    ctx_set_current: unsafe extern "C" fn(P) -> R,
    module_load_data: unsafe extern "C" fn(*mut P, *const c_void) -> R,
    module_get_function: unsafe extern "C" fn(*mut P, P, *const c_char) -> R,
    mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> R,
    mem_free: unsafe extern "C" fn(u64) -> R,
    memset_d8: unsafe extern "C" fn(u64, u8, usize) -> R,
    memcpy_htod: unsafe extern "C" fn(u64, *const c_void, usize) -> R,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, u64, usize) -> R,
    #[allow(clippy::type_complexity)]
    launch_kernel: unsafe extern "C" fn(
        P,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        P,
        *mut P,
        *mut P,
    ) -> R,
    stream_create: unsafe extern "C" fn(*mut P, c_uint) -> R,
    stream_synchronize: unsafe extern "C" fn(P) -> R,
    stream_begin_capture: unsafe extern "C" fn(P, c_int) -> R,
    stream_end_capture: unsafe extern "C" fn(P, *mut P) -> R,
    graph_instantiate: Instantiate,
    graph_launch: unsafe extern "C" fn(P, P) -> R,
    graph_exec_destroy: unsafe extern "C" fn(P) -> R,
    graph_destroy: unsafe extern "C" fn(P) -> R,
    get_error_string: unsafe extern "C" fn(R, *mut *const c_char) -> R,
}

impl Cu {
    fn check(&self, call: &str, code: R) -> Result<(), String> {
        if code == 0 {
            return Ok(());
        }
        let mut s = null();
        // SAFETY: `s` is a valid out-pointer; failure leaves it null.
        unsafe { (self.get_error_string)(code, &mut s) };
        Err(format!("{call}: {code} ({})", cstr(s)))
    }
}

struct Nvrtc {
    #[allow(clippy::type_complexity)]
    create_program: unsafe extern "C" fn(
        *mut P,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> R,
    compile_program: unsafe extern "C" fn(P, c_int, *const *const c_char) -> R,
    get_program_log_size: unsafe extern "C" fn(P, *mut usize) -> R,
    get_program_log: unsafe extern "C" fn(P, *mut c_char) -> R,
    get_ptx_size: unsafe extern "C" fn(P, *mut usize) -> R,
    get_ptx: unsafe extern "C" fn(P, *mut c_char) -> R,
    destroy_program: unsafe extern "C" fn(*mut P) -> R,
    get_error_string: unsafe extern "C" fn(R) -> *const c_char,
}

impl Nvrtc {
    fn check(&self, call: &str, code: R) -> Result<(), String> {
        if code == 0 {
            return Ok(());
        }
        // SAFETY: plain lookup of a static string.
        Err(format!(
            "{call}: {code} ({})",
            cstr(unsafe { (self.get_error_string)(code) })
        ))
    }

    /// Compiles `src` to NUL-terminated PTX with `opts`; the compile log is part of the error.
    fn ptx(&self, src: &str, opts: &[*const c_char]) -> Result<Vec<u8>, String> {
        let src = CString::new(src).map_err(|e| format!("nvrtcCreateProgram: {e}"))?;
        let mut prog = null_mut();
        // SAFETY: valid NUL-terminated strings; no headers.
        let code = unsafe {
            (self.create_program)(
                &mut prog,
                src.as_ptr(),
                c"gibbs.cu".as_ptr(),
                0,
                null(),
                null(),
            )
        };
        self.check("nvrtcCreateProgram", code)?;
        // SAFETY: `prog` is a live program until `destroy_program` below; buffers are sized by NVRTC.
        let out = (|| unsafe {
            let code = (self.compile_program)(prog, opts.len() as c_int, opts.as_ptr());
            if code != 0 {
                let mut n = 0;
                (self.get_program_log_size)(prog, &mut n);
                let mut log = vec![0u8; n.max(1)];
                (self.get_program_log)(prog, log.as_mut_ptr().cast());
                let log = String::from_utf8_lossy(&log);
                return Err(format!(
                    "{}\n{}",
                    self.check("nvrtcCompileProgram", code).unwrap_err(),
                    log.trim_end_matches('\0')
                ));
            }
            let mut n = 0;
            self.check("nvrtcGetPTXSize", (self.get_ptx_size)(prog, &mut n))?;
            let mut ptx = vec![0u8; n];
            self.check("nvrtcGetPTX", (self.get_ptx)(prog, ptx.as_mut_ptr().cast()))?;
            if ptx.last() != Some(&0) {
                ptx.push(0);
            }
            Ok(ptx)
        })();
        // SAFETY: `prog` was created above and is destroyed once.
        unsafe { (self.destroy_program)(&mut prog) };
        out
    }
}

/// CUDA device 0: its primary context and one blocking stream.
pub struct CudaApi {
    cu: Cu,
    rtc: Nvrtc,
    ctx: P,
    stream: P,
    /// `--gpu-architecture=compute_<major><minor>`.
    arch: CString,
    name: String,
    _libs: [DynLib; 2],
}

// SAFETY: the CUDA driver API is thread-safe; every method makes `ctx` current on the calling
// thread first, and `ctx`/`stream` are only passed to the driver, never dereferenced.
unsafe impl Send for CudaApi {}
unsafe impl Sync for CudaApi {}

impl CudaApi {
    fn open() -> Option<Self> {
        let lib = DynLib::open_any(&[
            "libcuda.so.1",
            "libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
            "libcuda.dylib",
        ])?;
        let nvrtc = DynLib::open_any(&["libnvrtc.so.12", "libnvrtc.so.11.2", "libnvrtc.so"])?;
        // SAFETY: every symbol is cast to its documented C signature.
        let (cu, rtc) = unsafe {
            let graph_instantiate = match sym(&lib, c"cuGraphInstantiateWithFlags") {
                Some(f) => Instantiate::WithFlags(f),
                None => Instantiate::V2(sym(&lib, c"cuGraphInstantiate_v2")?),
            };
            let cu = Cu {
                init: sym(&lib, c"cuInit")?,
                device_get_count: sym(&lib, c"cuDeviceGetCount")?,
                device_get: sym(&lib, c"cuDeviceGet")?,
                device_get_name: sym(&lib, c"cuDeviceGetName")?,
                device_get_attribute: sym(&lib, c"cuDeviceGetAttribute")?,
                primary_ctx_retain: sym(&lib, c"cuDevicePrimaryCtxRetain")?,
                ctx_set_current: sym(&lib, c"cuCtxSetCurrent")?,
                module_load_data: sym(&lib, c"cuModuleLoadData")?,
                module_get_function: sym(&lib, c"cuModuleGetFunction")?,
                mem_alloc: sym(&lib, c"cuMemAlloc_v2")?,
                mem_free: sym(&lib, c"cuMemFree_v2")?,
                memset_d8: sym(&lib, c"cuMemsetD8_v2")?,
                memcpy_htod: sym(&lib, c"cuMemcpyHtoD_v2")?,
                memcpy_dtoh: sym(&lib, c"cuMemcpyDtoH_v2")?,
                launch_kernel: sym(&lib, c"cuLaunchKernel")?,
                stream_create: sym(&lib, c"cuStreamCreate")?,
                stream_synchronize: sym(&lib, c"cuStreamSynchronize")?,
                stream_begin_capture: sym(&lib, c"cuStreamBeginCapture_v2")
                    .or_else(|| sym(&lib, c"cuStreamBeginCapture"))?,
                stream_end_capture: sym(&lib, c"cuStreamEndCapture")?,
                graph_instantiate,
                graph_launch: sym(&lib, c"cuGraphLaunch")?,
                graph_exec_destroy: sym(&lib, c"cuGraphExecDestroy")?,
                graph_destroy: sym(&lib, c"cuGraphDestroy")?,
                get_error_string: sym(&lib, c"cuGetErrorString")?,
            };
            let rtc = Nvrtc {
                create_program: sym(&nvrtc, c"nvrtcCreateProgram")?,
                compile_program: sym(&nvrtc, c"nvrtcCompileProgram")?,
                get_program_log_size: sym(&nvrtc, c"nvrtcGetProgramLogSize")?,
                get_program_log: sym(&nvrtc, c"nvrtcGetProgramLog")?,
                get_ptx_size: sym(&nvrtc, c"nvrtcGetPTXSize")?,
                get_ptx: sym(&nvrtc, c"nvrtcGetPTX")?,
                destroy_program: sym(&nvrtc, c"nvrtcDestroyProgram")?,
                get_error_string: sym(&nvrtc, c"nvrtcGetErrorString")?,
            };
            (cu, rtc)
        };
        let ok = |call: &str, code: R| cu.check(call, code).ok();
        let (mut count, mut dev, mut major, mut minor) = (0, 0, 0, 0);
        let mut name = [0 as c_char; 256];
        let (mut ctx, mut stream) = (null_mut(), null_mut());
        // SAFETY: valid out-pointers; `name` holds 255 chars plus NUL.
        unsafe {
            ok("cuInit", (cu.init)(0))?;
            ok("cuDeviceGetCount", (cu.device_get_count)(&mut count))?;
            if count < 1 {
                return None;
            }
            ok("cuDeviceGet", (cu.device_get)(&mut dev, 0))?;
            ok(
                "cuDeviceGetName",
                (cu.device_get_name)(name.as_mut_ptr(), 255, dev),
            )?;
            ok(
                "cuDeviceGetAttribute",
                (cu.device_get_attribute)(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                ),
            )?;
            ok(
                "cuDeviceGetAttribute",
                (cu.device_get_attribute)(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    dev,
                ),
            )?;
            ok(
                "cuDevicePrimaryCtxRetain",
                (cu.primary_ctx_retain)(&mut ctx, dev),
            )?;
            ok("cuCtxSetCurrent", (cu.ctx_set_current)(ctx))?;
            ok("cuStreamCreate", (cu.stream_create)(&mut stream, 0))?;
        }
        // SAFETY: `name` is NUL-terminated (zeroed, at most 255 chars written).
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .trim()
            .to_string();
        Some(Self {
            cu,
            rtc,
            ctx,
            stream,
            arch: CString::new(format!("--gpu-architecture=compute_{major}{minor}")).ok()?,
            name: if name.is_empty() {
                "NVIDIA CUDA (NVRTC)".into()
            } else {
                name
            },
            _libs: [lib, nvrtc],
        })
    }

    /// Makes the context current on the calling thread.
    fn current(&self) -> Result<(), String> {
        // SAFETY: `ctx` is the retained primary context.
        self.cu.check("cuCtxSetCurrent", unsafe {
            (self.cu.ctx_set_current)(self.ctx)
        })
    }
}

impl GpuApi for CudaApi {
    fn device_name(&self) -> &str {
        &self.name
    }

    fn dialect(&self) -> Dialect {
        Dialect::Cuda
    }

    fn alloc(&self, bytes: usize) -> Result<Buf, String> {
        self.current()?;
        let mut p = 0u64;
        // SAFETY: valid out-pointer.
        self.cu.check("cuMemAlloc_v2", unsafe {
            (self.cu.mem_alloc)(&mut p, bytes)
        })?;
        if p == 0 {
            return Err("cuMemAlloc_v2: returned a null pointer".into());
        }
        // SAFETY: `p` holds `bytes` bytes.
        if let Err(e) = self
            .cu
            .check("cuMemsetD8_v2", unsafe { (self.cu.memset_d8)(p, 0, bytes) })
        {
            // SAFETY: `p` was allocated above.
            unsafe { (self.cu.mem_free)(p) };
            return Err(e);
        }
        Ok(Buf(p))
    }

    fn free(&self, buf: Buf) {
        if self.current().is_ok() {
            // SAFETY: `buf` came from `alloc`.
            unsafe { (self.cu.mem_free)(buf.0) };
        }
    }

    fn upload(&self, buf: Buf, offset: usize, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        self.current()?;
        // SAFETY: the caller keeps `offset + data.len()` within `buf`. The legacy-stream copy is
        // ordered with the blocking stream.
        let code = unsafe {
            (self.cu.memcpy_htod)(buf.0 + offset as u64, data.as_ptr().cast(), data.len())
        };
        self.cu.check("cuMemcpyHtoD_v2", code)
    }

    fn download(&self, buf: Buf, offset: usize, out: &mut [u8]) -> Result<(), String> {
        if out.is_empty() {
            return Ok(());
        }
        self.sync()?;
        // SAFETY: the caller keeps `offset + out.len()` within `buf`; the copy is synchronous.
        let code = unsafe {
            (self.cu.memcpy_dtoh)(out.as_mut_ptr().cast(), buf.0 + offset as u64, out.len())
        };
        self.cu.check("cuMemcpyDtoH_v2", code)
    }

    fn compile(&self, src: &str) -> Result<Module, String> {
        // No fma contraction: codec math must round like the reference decoders.
        let opts = [
            self.arch.as_ptr(),
            c"-default-device".as_ptr(),
            c"--fmad=false".as_ptr(),
        ];
        let ptx = self.rtc.ptx(src, &opts)?;
        self.current()?;
        let mut m = null_mut();
        // SAFETY: `ptx` is NUL-terminated PTX text.
        self.cu.check("cuModuleLoadData", unsafe {
            (self.cu.module_load_data)(&mut m, ptx.as_ptr().cast())
        })?;
        Ok(Module(m as u64))
    }

    fn kernel(&self, module: &Module, name: &str) -> Result<Kernel, String> {
        self.current()?;
        let cname = CString::new(name).map_err(|e| format!("cuModuleGetFunction: {e}"))?;
        let mut f = null_mut();
        // SAFETY: `module` came from `compile`.
        let code = unsafe { (self.cu.module_get_function)(&mut f, module.0 as P, cname.as_ptr()) };
        self.cu
            .check("cuModuleGetFunction", code)
            .map_err(|e| format!("{e}: {name}"))?;
        Ok(Kernel(f as u64))
    }

    fn launch(
        &self,
        k: Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[Arg],
    ) -> Result<(), String> {
        if args.len() > MAX_ARGS {
            return Err(format!(
                "cuLaunchKernel: {} args exceed {MAX_ARGS}",
                args.len()
            ));
        }
        self.current()?;
        // One u64 slot per argument; a 4-byte parameter reads the low half (little-endian hosts).
        let mut vals = [0u64; MAX_ARGS];
        let mut ptrs = [null_mut::<c_void>(); MAX_ARGS];
        for ((a, v), p) in args.iter().zip(&mut vals).zip(&mut ptrs) {
            *v = match *a {
                Arg::Buf(b) => b.0,
                Arg::U32(x) => x as u64,
                Arg::F32(x) => x.to_bits() as u64,
                Arg::U64(x) => x,
            };
            *p = (v as *mut u64).cast();
        }
        // SAFETY: `ptrs[..args.len()]` point at live values matching the kernel's parameters.
        let code = unsafe {
            (self.cu.launch_kernel)(
                k.0 as P,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                0,
                self.stream,
                ptrs.as_mut_ptr(),
                null_mut(),
            )
        };
        self.cu.check("cuLaunchKernel", code)
    }

    fn sync(&self) -> Result<(), String> {
        self.current()?;
        // SAFETY: `stream` is live.
        self.cu.check("cuStreamSynchronize", unsafe {
            (self.cu.stream_synchronize)(self.stream)
        })
    }

    fn capture_begin(&self) -> Result<bool, String> {
        self.current()?;
        // SAFETY: `stream` is live.
        let code =
            unsafe { (self.cu.stream_begin_capture)(self.stream, CU_STREAM_CAPTURE_MODE_GLOBAL) };
        self.cu.check("cuStreamBeginCapture_v2", code)?;
        Ok(true)
    }

    fn capture_end(&self) -> Result<Replay, String> {
        self.current()?;
        let (mut graph, mut exec) = (null_mut(), null_mut());
        // SAFETY: valid out-pointers; `graph` is destroyed once after instantiation.
        unsafe {
            self.cu.check(
                "cuStreamEndCapture",
                (self.cu.stream_end_capture)(self.stream, &mut graph),
            )?;
            let r = match self.cu.graph_instantiate {
                Instantiate::WithFlags(f) => self
                    .cu
                    .check("cuGraphInstantiateWithFlags", f(&mut exec, graph, 0)),
                Instantiate::V2(f) => self.cu.check(
                    "cuGraphInstantiate_v2",
                    f(&mut exec, graph, null_mut(), null_mut(), 0),
                ),
            };
            (self.cu.graph_destroy)(graph);
            r?;
        }
        Ok(Replay(exec as u64))
    }

    fn replay(&self, r: Replay) -> Result<(), String> {
        self.current()?;
        // SAFETY: `r` came from `capture_end`.
        self.cu.check("cuGraphLaunch", unsafe {
            (self.cu.graph_launch)(r.0 as P, self.stream)
        })
    }

    fn replay_free(&self, r: Replay) {
        if self.current().is_ok() {
            // SAFETY: `r` came from `capture_end`.
            unsafe { (self.cu.graph_exec_destroy)(r.0 as P) };
        }
    }
}

/// The process-wide CUDA API: device 0, once its libraries load and the probe kernel passes.
pub fn api() -> Option<Arc<dyn GpuApi>> {
    static API: LazyLock<Option<Arc<dyn GpuApi>>> = LazyLock::new(|| {
        let api: Arc<dyn GpuApi> = Arc::new(CudaApi::open()?);
        portable::probe(&*api).ok()?;
        Some(api)
    });
    API.clone()
}

pub fn is_available() -> bool {
    api().is_some()
}

pub fn device_name() -> String {
    api().map_or_else(
        || "NVIDIA CUDA (NVRTC)".into(),
        |a| a.device_name().to_string(),
    )
}

pub fn supports_op(op: &Op, dtype: DType) -> bool {
    portable::supports_op(op, dtype)
}

pub fn bandwidth_gbps() -> Option<f64> {
    api().and_then(|a| portable::bandwidth_gbps(&*a))
}
