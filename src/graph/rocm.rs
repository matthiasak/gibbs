//! AMD ROCm backend: [`HipApi`] implements the portable [`GpuApi`] over the HIP runtime
//! (`libamdhip64`) and hipRTC (`libhiprtc`), both loaded at runtime through [`DynLib`] (no
//! build-time ROCm SDK). The portable kernel source compiles to a code object for the device's
//! `gfx` architecture; every launch goes to one blocking stream, which stream capture records
//! into HIP graphs.

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
/// Bytes handed to `hipGetDeviceProperties`: larger than every ROCm `hipDeviceProp_t` layout.
const PROPS_BYTES: usize = 8192;
const HIP_STREAM_CAPTURE_MODE_GLOBAL: c_int = 0;

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
    // SAFETY: HIP/hipRTC error strings are static NUL-terminated strings.
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// The gcnArchName of a raw `hipDeviceProp_t`: the first NUL-terminated ASCII string in it that
/// starts with `gfx` (its offset differs across ROCm versions).
fn gfx_arch(props: &[u8]) -> Option<String> {
    (0..props.len()).find_map(|i| {
        if !props[i..].starts_with(b"gfx") || (i > 0 && props[i - 1] != 0) {
            return None;
        }
        let s = CStr::from_bytes_until_nul(&props[i..])
            .ok()?
            .to_str()
            .ok()?;
        s.bytes()
            .all(|b| b.is_ascii_graphic())
            .then(|| s.to_string())
    })
}

struct Hip {
    init: unsafe extern "C" fn(c_uint) -> R,
    get_device_count: unsafe extern "C" fn(*mut c_int) -> R,
    set_device: unsafe extern "C" fn(c_int) -> R,
    get_device_properties: unsafe extern "C" fn(*mut c_void, c_int) -> R,
    module_load_data: unsafe extern "C" fn(*mut P, *const c_void) -> R,
    module_get_function: unsafe extern "C" fn(*mut P, P, *const c_char) -> R,
    malloc: unsafe extern "C" fn(*mut P, usize) -> R,
    free: unsafe extern "C" fn(P) -> R,
    memset: unsafe extern "C" fn(P, c_int, usize) -> R,
    memcpy_htod: unsafe extern "C" fn(P, *const c_void, usize) -> R,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, P, usize) -> R,
    #[allow(clippy::type_complexity)]
    module_launch_kernel: unsafe extern "C" fn(
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
    stream_create: unsafe extern "C" fn(*mut P) -> R,
    stream_synchronize: unsafe extern "C" fn(P) -> R,
    stream_begin_capture: unsafe extern "C" fn(P, c_int) -> R,
    stream_end_capture: unsafe extern "C" fn(P, *mut P) -> R,
    graph_instantiate: unsafe extern "C" fn(*mut P, P, *mut P, *mut c_char, usize) -> R,
    graph_launch: unsafe extern "C" fn(P, P) -> R,
    graph_exec_destroy: unsafe extern "C" fn(P) -> R,
    graph_destroy: unsafe extern "C" fn(P) -> R,
    get_error_string: unsafe extern "C" fn(R) -> *const c_char,
}

impl Hip {
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
}

struct HipRtc {
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
    get_code_size: unsafe extern "C" fn(P, *mut usize) -> R,
    get_code: unsafe extern "C" fn(P, *mut c_char) -> R,
    destroy_program: unsafe extern "C" fn(*mut P) -> R,
    get_error_string: unsafe extern "C" fn(R) -> *const c_char,
}

impl HipRtc {
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

    /// Compiles `src` to a code object with `opts`; the compile log is part of the error.
    fn code(&self, src: &str, opts: &[*const c_char]) -> Result<Vec<u8>, String> {
        let src = CString::new(src).map_err(|e| format!("hiprtcCreateProgram: {e}"))?;
        let mut prog = null_mut();
        // SAFETY: valid NUL-terminated strings; no headers.
        let code = unsafe {
            (self.create_program)(
                &mut prog,
                src.as_ptr(),
                c"gibbs.hip".as_ptr(),
                0,
                null(),
                null(),
            )
        };
        self.check("hiprtcCreateProgram", code)?;
        // SAFETY: `prog` is a live program until `destroy_program` below; buffers are sized by hipRTC.
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
                    self.check("hiprtcCompileProgram", code).unwrap_err(),
                    log.trim_end_matches('\0')
                ));
            }
            let mut n = 0;
            self.check("hiprtcGetCodeSize", (self.get_code_size)(prog, &mut n))?;
            let mut code = vec![0u8; n];
            self.check(
                "hiprtcGetCode",
                (self.get_code)(prog, code.as_mut_ptr().cast()),
            )?;
            Ok(code)
        })();
        // SAFETY: `prog` was created above and is destroyed once.
        unsafe { (self.destroy_program)(&mut prog) };
        out
    }
}

/// HIP device 0 and one blocking stream.
pub struct HipApi {
    hip: Hip,
    rtc: HipRtc,
    stream: P,
    /// `--offload-arch=<gcnArchName>`, when the device properties name one.
    arch: Option<CString>,
    name: String,
    _libs: [DynLib; 2],
}

// SAFETY: the HIP runtime is thread-safe; every method selects device 0 on the calling thread
// first, and `stream` is only passed to the runtime, never dereferenced.
unsafe impl Send for HipApi {}
unsafe impl Sync for HipApi {}

impl HipApi {
    fn open() -> Option<Self> {
        let lib = DynLib::open_any(&[
            "libamdhip64.so.6",
            "libamdhip64.so.5",
            "libamdhip64.so",
            "/opt/rocm/lib/libamdhip64.so",
        ])?;
        let hiprtc = DynLib::open_any(&[
            "libhiprtc.so.6",
            "libhiprtc.so.5",
            "libhiprtc.so",
            "/opt/rocm/lib/libhiprtc.so",
        ])?;
        // SAFETY: every symbol is cast to its documented C signature.
        let (hip, rtc) = unsafe {
            let hip = Hip {
                init: sym(&lib, c"hipInit")?,
                get_device_count: sym(&lib, c"hipGetDeviceCount")?,
                set_device: sym(&lib, c"hipSetDevice")?,
                get_device_properties: sym(&lib, c"hipGetDevicePropertiesR0600")
                    .or_else(|| sym(&lib, c"hipGetDeviceProperties"))?,
                module_load_data: sym(&lib, c"hipModuleLoadData")?,
                module_get_function: sym(&lib, c"hipModuleGetFunction")?,
                malloc: sym(&lib, c"hipMalloc")?,
                free: sym(&lib, c"hipFree")?,
                memset: sym(&lib, c"hipMemset")?,
                memcpy_htod: sym(&lib, c"hipMemcpyHtoD")?,
                memcpy_dtoh: sym(&lib, c"hipMemcpyDtoH")?,
                module_launch_kernel: sym(&lib, c"hipModuleLaunchKernel")?,
                stream_create: sym(&lib, c"hipStreamCreate")?,
                stream_synchronize: sym(&lib, c"hipStreamSynchronize")?,
                stream_begin_capture: sym(&lib, c"hipStreamBeginCapture")?,
                stream_end_capture: sym(&lib, c"hipStreamEndCapture")?,
                graph_instantiate: sym(&lib, c"hipGraphInstantiate")?,
                graph_launch: sym(&lib, c"hipGraphLaunch")?,
                graph_exec_destroy: sym(&lib, c"hipGraphExecDestroy")?,
                graph_destroy: sym(&lib, c"hipGraphDestroy")?,
                get_error_string: sym(&lib, c"hipGetErrorString")?,
            };
            let rtc = HipRtc {
                create_program: sym(&hiprtc, c"hiprtcCreateProgram")?,
                compile_program: sym(&hiprtc, c"hiprtcCompileProgram")?,
                get_program_log_size: sym(&hiprtc, c"hiprtcGetProgramLogSize")?,
                get_program_log: sym(&hiprtc, c"hiprtcGetProgramLog")?,
                get_code_size: sym(&hiprtc, c"hiprtcGetCodeSize")?,
                get_code: sym(&hiprtc, c"hiprtcGetCode")?,
                destroy_program: sym(&hiprtc, c"hiprtcDestroyProgram")?,
                get_error_string: sym(&hiprtc, c"hiprtcGetErrorString")?,
            };
            (hip, rtc)
        };
        let ok = |call: &str, code: R| hip.check(call, code).ok();
        let mut count = 0;
        let mut props = vec![0u8; PROPS_BYTES];
        let mut stream = null_mut();
        // SAFETY: valid out-pointers; `props` is larger than any `hipDeviceProp_t`.
        unsafe {
            ok("hipInit", (hip.init)(0))?;
            ok("hipGetDeviceCount", (hip.get_device_count)(&mut count))?;
            if count < 1 {
                return None;
            }
            ok("hipSetDevice", (hip.set_device)(0))?;
            ok(
                "hipGetDeviceProperties",
                (hip.get_device_properties)(props.as_mut_ptr().cast(), 0),
            )?;
            ok("hipStreamCreate", (hip.stream_create)(&mut stream))?;
        }
        let name = CStr::from_bytes_until_nul(&props)
            .map_or(String::new(), |s| s.to_string_lossy().trim().to_string());
        Some(Self {
            hip,
            rtc,
            stream,
            arch: gfx_arch(&props).and_then(|a| CString::new(format!("--offload-arch={a}")).ok()),
            name: if name.is_empty() {
                "AMD ROCm (HIP)".into()
            } else {
                name
            },
            _libs: [lib, hiprtc],
        })
    }

    /// Selects device 0 on the calling thread.
    fn current(&self) -> Result<(), String> {
        // SAFETY: plain runtime call.
        self.hip
            .check("hipSetDevice", unsafe { (self.hip.set_device)(0) })
    }
}

impl GpuApi for HipApi {
    fn device_name(&self) -> &str {
        &self.name
    }

    fn dialect(&self) -> Dialect {
        Dialect::Hip
    }

    fn alloc(&self, bytes: usize) -> Result<Buf, String> {
        self.current()?;
        let mut p = null_mut();
        // SAFETY: valid out-pointer.
        self.hip
            .check("hipMalloc", unsafe { (self.hip.malloc)(&mut p, bytes) })?;
        if p.is_null() {
            return Err("hipMalloc: returned a null pointer".into());
        }
        // SAFETY: `p` holds `bytes` bytes.
        if let Err(e) = self
            .hip
            .check("hipMemset", unsafe { (self.hip.memset)(p, 0, bytes) })
        {
            // SAFETY: `p` was allocated above.
            unsafe { (self.hip.free)(p) };
            return Err(e);
        }
        Ok(Buf(p as u64))
    }

    fn free(&self, buf: Buf) {
        if self.current().is_ok() {
            // SAFETY: `buf` came from `alloc`.
            unsafe { (self.hip.free)(buf.0 as P) };
        }
    }

    fn upload(&self, buf: Buf, offset: usize, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        self.current()?;
        // SAFETY: the caller keeps `offset + data.len()` within `buf`. The null-stream copy is
        // ordered with the blocking stream.
        let code = unsafe {
            (self.hip.memcpy_htod)(
                (buf.0 + offset as u64) as P,
                data.as_ptr().cast(),
                data.len(),
            )
        };
        self.hip.check("hipMemcpyHtoD", code)
    }

    fn download(&self, buf: Buf, offset: usize, out: &mut [u8]) -> Result<(), String> {
        if out.is_empty() {
            return Ok(());
        }
        self.sync()?;
        // SAFETY: the caller keeps `offset + out.len()` within `buf`; the copy is synchronous.
        let code = unsafe {
            (self.hip.memcpy_dtoh)(
                out.as_mut_ptr().cast(),
                (buf.0 + offset as u64) as P,
                out.len(),
            )
        };
        self.hip.check("hipMemcpyDtoH", code)
    }

    fn compile(&self, src: &str) -> Result<Module, String> {
        // No fma contraction: codec math must round like the reference decoders.
        let mut opts = [c"-ffp-contract=off".as_ptr(), null()];
        let n = match &self.arch {
            Some(a) => {
                opts[1] = a.as_ptr();
                2
            }
            None => 1,
        };
        let code = self.rtc.code(src, &opts[..n])?;
        self.current()?;
        let mut m = null_mut();
        // SAFETY: `code` is a complete code object.
        self.hip.check("hipModuleLoadData", unsafe {
            (self.hip.module_load_data)(&mut m, code.as_ptr().cast())
        })?;
        Ok(Module(m as u64))
    }

    fn kernel(&self, module: &Module, name: &str) -> Result<Kernel, String> {
        self.current()?;
        let cname = CString::new(name).map_err(|e| format!("hipModuleGetFunction: {e}"))?;
        let mut f = null_mut();
        // SAFETY: `module` came from `compile`.
        let code = unsafe { (self.hip.module_get_function)(&mut f, module.0 as P, cname.as_ptr()) };
        self.hip
            .check("hipModuleGetFunction", code)
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
                "hipModuleLaunchKernel: {} args exceed {MAX_ARGS}",
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
            (self.hip.module_launch_kernel)(
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
        self.hip.check("hipModuleLaunchKernel", code)
    }

    fn sync(&self) -> Result<(), String> {
        self.current()?;
        // SAFETY: `stream` is live.
        self.hip.check("hipStreamSynchronize", unsafe {
            (self.hip.stream_synchronize)(self.stream)
        })
    }

    fn capture_begin(&self) -> Result<bool, String> {
        self.current()?;
        // SAFETY: `stream` is live.
        let code =
            unsafe { (self.hip.stream_begin_capture)(self.stream, HIP_STREAM_CAPTURE_MODE_GLOBAL) };
        self.hip.check("hipStreamBeginCapture", code)?;
        Ok(true)
    }

    fn capture_end(&self) -> Result<Replay, String> {
        self.current()?;
        let (mut graph, mut exec) = (null_mut(), null_mut());
        // SAFETY: valid out-pointers; `graph` is destroyed once after instantiation.
        unsafe {
            self.hip.check(
                "hipStreamEndCapture",
                (self.hip.stream_end_capture)(self.stream, &mut graph),
            )?;
            let code = (self.hip.graph_instantiate)(&mut exec, graph, null_mut(), null_mut(), 0);
            (self.hip.graph_destroy)(graph);
            self.hip.check("hipGraphInstantiate", code)?;
        }
        Ok(Replay(exec as u64))
    }

    fn replay(&self, r: Replay) -> Result<(), String> {
        self.current()?;
        // SAFETY: `r` came from `capture_end`.
        self.hip.check("hipGraphLaunch", unsafe {
            (self.hip.graph_launch)(r.0 as P, self.stream)
        })
    }

    fn replay_free(&self, r: Replay) {
        if self.current().is_ok() {
            // SAFETY: `r` came from `capture_end`.
            unsafe { (self.hip.graph_exec_destroy)(r.0 as P) };
        }
    }
}

/// The process-wide HIP API: device 0, once its libraries load and the probe kernel passes.
pub fn api() -> Option<Arc<dyn GpuApi>> {
    static API: LazyLock<Option<Arc<dyn GpuApi>>> = LazyLock::new(|| {
        let api: Arc<dyn GpuApi> = Arc::new(HipApi::open()?);
        portable::probe(&*api).ok()?;
        Some(api)
    });
    API.clone()
}

pub fn is_available() -> bool {
    api().is_some()
}

pub fn device_name() -> String {
    api().map_or_else(|| "AMD ROCm (HIP)".into(), |a| a.device_name().to_string())
}

pub fn supports_op(op: &Op, dtype: DType) -> bool {
    portable::supports_op(op, dtype)
}

pub fn bandwidth_gbps() -> Option<f64> {
    api().and_then(|a| portable::bandwidth_gbps(&*a))
}
