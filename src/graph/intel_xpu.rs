//! Intel XPU backend: an OpenCL 1.2 [`GpuApi`] (`libOpenCL.so.1`, or `OpenCL.framework` on
//! macOS, loaded at runtime through [`DynLib`]) running the portable kernels.
//!
//! The platform is the first whose name contains `Intel`; `GIBBS_XPU_ANY_OPENCL=1` accepts any
//! platform (on Apple Silicon that is the Apple GPU, which is how this backend is verified there).
//! The queue is in order and every copy is blocking, so no work is in flight across runs.

use crate::graph::plan::DynLib;
use crate::graph::portable::api::{Arg, Buf, GpuApi, Kernel, Module};
use crate::graph::portable::dialect::Dialect;
use crate::graph::{DType, Op};
use std::ffi::{c_char, c_void, CString};
use std::sync::{Arc, LazyLock, Mutex};

type ClInt = i32;
type Handle = *mut c_void;

const CL_PLATFORM_NAME: u32 = 0x0902;
const CL_DEVICE_TYPE_GPU: u64 = 1 << 2;
const CL_DEVICE_NAME: u32 = 0x102B;
const CL_MEM_READ_WRITE: u64 = 1;
const CL_PROGRAM_BUILD_LOG: u32 = 0x1183;
const CL_TRUE: u32 = 1;

#[allow(clippy::type_complexity)]
struct Fns {
    get_platform_ids: unsafe extern "C" fn(u32, *mut Handle, *mut u32) -> ClInt,
    get_platform_info: unsafe extern "C" fn(Handle, u32, usize, *mut c_void, *mut usize) -> ClInt,
    get_device_ids: unsafe extern "C" fn(Handle, u64, u32, *mut Handle, *mut u32) -> ClInt,
    get_device_info: unsafe extern "C" fn(Handle, u32, usize, *mut c_void, *mut usize) -> ClInt,
    create_context: unsafe extern "C" fn(
        *const isize,
        u32,
        *const Handle,
        Option<unsafe extern "C" fn()>,
        *mut c_void,
        *mut ClInt,
    ) -> Handle,
    create_command_queue: unsafe extern "C" fn(Handle, Handle, u64, *mut ClInt) -> Handle,
    create_buffer: unsafe extern "C" fn(Handle, u64, usize, *mut c_void, *mut ClInt) -> Handle,
    release_mem_object: unsafe extern "C" fn(Handle) -> ClInt,
    enqueue_write_buffer: unsafe extern "C" fn(
        Handle,
        Handle,
        u32,
        usize,
        usize,
        *const c_void,
        u32,
        *const Handle,
        *mut Handle,
    ) -> ClInt,
    enqueue_read_buffer: unsafe extern "C" fn(
        Handle,
        Handle,
        u32,
        usize,
        usize,
        *mut c_void,
        u32,
        *const Handle,
        *mut Handle,
    ) -> ClInt,
    create_program_with_source:
        unsafe extern "C" fn(Handle, u32, *const *const c_char, *const usize, *mut ClInt) -> Handle,
    build_program: unsafe extern "C" fn(
        Handle,
        u32,
        *const Handle,
        *const c_char,
        Option<unsafe extern "C" fn()>,
        *mut c_void,
    ) -> ClInt,
    get_program_build_info:
        unsafe extern "C" fn(Handle, Handle, u32, usize, *mut c_void, *mut usize) -> ClInt,
    create_kernel: unsafe extern "C" fn(Handle, *const c_char, *mut ClInt) -> Handle,
    set_kernel_arg: unsafe extern "C" fn(Handle, u32, usize, *const c_void) -> ClInt,
    enqueue_nd_range_kernel: unsafe extern "C" fn(
        Handle,
        Handle,
        u32,
        *const usize,
        *const usize,
        *const usize,
        u32,
        *const Handle,
        *mut Handle,
    ) -> ClInt,
    finish: unsafe extern "C" fn(Handle) -> ClInt,
}

/// One OpenCL GPU device with its context and in-order queue.
pub struct OpenClApi {
    _lib: DynLib,
    f: Fns,
    device: Handle,
    context: Handle,
    queue: Handle,
    name: String,
    /// `clSetKernelArg` + enqueue must not interleave across threads on one kernel object.
    launch: Mutex<()>,
}

// SAFETY: OpenCL objects are thread-safe except kernel argument setting, which `launch` serializes.
unsafe impl Send for OpenClApi {}
unsafe impl Sync for OpenClApi {}

fn check(call: &str, code: ClInt) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{call}: OpenCL error {code}"))
    }
}

fn info_string(
    f: unsafe extern "C" fn(Handle, u32, usize, *mut c_void, *mut usize) -> ClInt,
    h: Handle,
    param: u32,
) -> String {
    let mut buf = [0u8; 256];
    let mut len = 0usize;
    // SAFETY: `buf` holds 256 bytes; OpenCL writes at most that many.
    let code = unsafe { f(h, param, buf.len(), buf.as_mut_ptr().cast(), &mut len) };
    if code != 0 {
        return String::new();
    }
    let end = buf
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(len.min(buf.len()));
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

impl OpenClApi {
    fn open() -> Result<Self, String> {
        let lib = DynLib::open_any(&[
            "libOpenCL.so.1",
            "libOpenCL.so",
            "/System/Library/Frameworks/OpenCL.framework/OpenCL",
        ])
        .ok_or("libOpenCL not found")?;
        macro_rules! sym {
            ($name:literal) => {
                // SAFETY: the symbol is the OpenCL 1.2 entry point of the declared signature.
                unsafe {
                    std::mem::transmute(
                        lib.sym_ptr(
                            std::ffi::CStr::from_bytes_with_nul(concat!($name, "\0").as_bytes())
                                .expect("literal"),
                        )
                        .ok_or(concat!("missing OpenCL symbol ", $name))?,
                    )
                }
            };
        }
        let f = Fns {
            get_platform_ids: sym!("clGetPlatformIDs"),
            get_platform_info: sym!("clGetPlatformInfo"),
            get_device_ids: sym!("clGetDeviceIDs"),
            get_device_info: sym!("clGetDeviceInfo"),
            create_context: sym!("clCreateContext"),
            create_command_queue: sym!("clCreateCommandQueue"),
            create_buffer: sym!("clCreateBuffer"),
            release_mem_object: sym!("clReleaseMemObject"),
            enqueue_write_buffer: sym!("clEnqueueWriteBuffer"),
            enqueue_read_buffer: sym!("clEnqueueReadBuffer"),
            create_program_with_source: sym!("clCreateProgramWithSource"),
            build_program: sym!("clBuildProgram"),
            get_program_build_info: sym!("clGetProgramBuildInfo"),
            create_kernel: sym!("clCreateKernel"),
            set_kernel_arg: sym!("clSetKernelArg"),
            enqueue_nd_range_kernel: sym!("clEnqueueNDRangeKernel"),
            finish: sym!("clFinish"),
        };
        let any = std::env::var("GIBBS_XPU_ANY_OPENCL").as_deref() == Ok("1");
        let mut n = 0u32;
        // SAFETY: count query, then a buffer of exactly `n` platform ids.
        check("clGetPlatformIDs", unsafe {
            (f.get_platform_ids)(0, std::ptr::null_mut(), &mut n)
        })?;
        let mut plats = vec![std::ptr::null_mut(); n as usize];
        check("clGetPlatformIDs", unsafe {
            (f.get_platform_ids)(n, plats.as_mut_ptr(), &mut n)
        })?;
        for p in plats {
            let pname = info_string(f.get_platform_info, p, CL_PLATFORM_NAME);
            if !any && !pname.contains("Intel") {
                continue;
            }
            let mut dev: Handle = std::ptr::null_mut();
            let mut nd = 0u32;
            // SAFETY: room for one device id.
            if unsafe { (f.get_device_ids)(p, CL_DEVICE_TYPE_GPU, 1, &mut dev, &mut nd) } != 0
                || nd == 0
            {
                continue;
            }
            let name = info_string(f.get_device_info, dev, CL_DEVICE_NAME);
            let mut err = 0;
            // SAFETY: one valid device id, no properties or callback.
            let context = unsafe {
                (f.create_context)(
                    std::ptr::null(),
                    1,
                    &dev,
                    None,
                    std::ptr::null_mut(),
                    &mut err,
                )
            };
            check("clCreateContext", err)?;
            // SAFETY: valid context and device; in-order queue (no properties).
            let queue = unsafe { (f.create_command_queue)(context, dev, 0, &mut err) };
            check("clCreateCommandQueue", err)?;
            return Ok(Self {
                _lib: lib,
                f,
                device: dev,
                context,
                queue,
                name,
                launch: Mutex::new(()),
            });
        }
        Err("no matching OpenCL GPU platform".into())
    }
}

impl GpuApi for OpenClApi {
    fn device_name(&self) -> &str {
        &self.name
    }

    fn dialect(&self) -> Dialect {
        Dialect::OpenCl
    }

    fn alloc(&self, bytes: usize) -> Result<Buf, String> {
        let mut err = 0;
        // SAFETY: plain device allocation, no host pointer.
        let mem = unsafe {
            (self.f.create_buffer)(
                self.context,
                CL_MEM_READ_WRITE,
                bytes,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        check("clCreateBuffer", err)?;
        // Zeroed by blocking writes of a zero chunk: Apple's `clEnqueueFillBuffer` fills a buffer
        // that was never written with garbage instead of the pattern.
        let zero = vec![0u8; bytes.min(1 << 22)];
        let buf = Buf(mem as u64);
        for off in (0..bytes).step_by(zero.len()) {
            if let Err(e) = self.upload(buf, off, &zero[..zero.len().min(bytes - off)]) {
                self.free(buf);
                return Err(e);
            }
        }
        Ok(buf)
    }

    fn free(&self, buf: Buf) {
        // SAFETY: `buf` came from `alloc`; the in-order queue finishes users before release.
        unsafe { (self.f.release_mem_object)(buf.0 as Handle) };
    }

    fn upload(&self, buf: Buf, offset: usize, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: blocking write of `data` into a buffer the caller sized.
        check("clEnqueueWriteBuffer", unsafe {
            (self.f.enqueue_write_buffer)(
                self.queue,
                buf.0 as Handle,
                CL_TRUE,
                offset,
                data.len(),
                data.as_ptr().cast(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        })
    }

    fn download(&self, buf: Buf, offset: usize, out: &mut [u8]) -> Result<(), String> {
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: blocking read into `out`.
        check("clEnqueueReadBuffer", unsafe {
            (self.f.enqueue_read_buffer)(
                self.queue,
                buf.0 as Handle,
                CL_TRUE,
                offset,
                out.len(),
                out.as_mut_ptr().cast(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        })
    }

    fn compile(&self, src: &str) -> Result<Module, String> {
        let mut err = 0;
        let ptr = src.as_ptr() as *const c_char;
        let len = src.len();
        // SAFETY: one source string of `len` bytes.
        let prog =
            unsafe { (self.f.create_program_with_source)(self.context, 1, &ptr, &len, &mut err) };
        check("clCreateProgramWithSource", err)?;
        let opts = CString::new("-cl-std=CL1.2").expect("no NUL");
        // SAFETY: valid program and device; synchronous build.
        let code = unsafe {
            (self.f.build_program)(
                prog,
                1,
                &self.device,
                opts.as_ptr(),
                None,
                std::ptr::null_mut(),
            )
        };
        if code != 0 {
            let mut size = 0usize;
            // SAFETY: log size query, then a buffer of that size.
            unsafe {
                (self.f.get_program_build_info)(
                    prog,
                    self.device,
                    CL_PROGRAM_BUILD_LOG,
                    0,
                    std::ptr::null_mut(),
                    &mut size,
                )
            };
            let mut log = vec![0u8; size];
            unsafe {
                (self.f.get_program_build_info)(
                    prog,
                    self.device,
                    CL_PROGRAM_BUILD_LOG,
                    size,
                    log.as_mut_ptr().cast(),
                    std::ptr::null_mut(),
                )
            };
            return Err(format!(
                "clBuildProgram: OpenCL error {code}\n{}",
                String::from_utf8_lossy(&log).trim_end_matches('\0')
            ));
        }
        Ok(Module(prog as u64))
    }

    fn kernel(&self, module: &Module, name: &str) -> Result<Kernel, String> {
        let cname = CString::new(name).map_err(|e| e.to_string())?;
        let mut err = 0;
        // SAFETY: built program, NUL-terminated kernel name.
        let k = unsafe { (self.f.create_kernel)(module.0 as Handle, cname.as_ptr(), &mut err) };
        check(&format!("clCreateKernel({name})"), err)?;
        Ok(Kernel(k as u64))
    }

    fn launch(
        &self,
        k: Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[Arg],
    ) -> Result<(), String> {
        let _guard = self.launch.lock().unwrap_or_else(|e| e.into_inner());
        let kh = k.0 as Handle;
        for (i, a) in args.iter().enumerate() {
            let (size, ptr): (usize, *const c_void) = match a {
                Arg::Buf(b) => (8, (&b.0 as *const u64).cast()),
                Arg::U32(v) => (4, (v as *const u32).cast()),
                Arg::F32(v) => (4, (v as *const f32).cast()),
                Arg::U64(v) => (8, (v as *const u64).cast()),
            };
            // SAFETY: `ptr` points at `size` bytes of the argument value (cl_mem is 8 bytes).
            check(&format!("clSetKernelArg({i})"), unsafe {
                (self.f.set_kernel_arg)(kh, i as u32, size, ptr)
            })?;
        }
        let global: [usize; 3] = std::array::from_fn(|d| grid[d] as usize * block[d] as usize);
        let local: [usize; 3] = std::array::from_fn(|d| block[d] as usize);
        // SAFETY: 3-D range with matching global/local arrays.
        check("clEnqueueNDRangeKernel", unsafe {
            (self.f.enqueue_nd_range_kernel)(
                self.queue,
                kh,
                3,
                std::ptr::null(),
                global.as_ptr(),
                local.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        })
    }

    fn sync(&self) -> Result<(), String> {
        // SAFETY: valid queue.
        check("clFinish", unsafe { (self.f.finish)(self.queue) })
    }
}

/// The process's OpenCL device when it opens and passes the portable probe kernel.
static API: LazyLock<Option<Arc<dyn GpuApi>>> = LazyLock::new(|| {
    let api: Arc<dyn GpuApi> = Arc::new(OpenClApi::open().ok()?);
    crate::graph::portable::probe(&*api).ok()?;
    Some(api)
});

pub fn api() -> Option<Arc<dyn GpuApi>> {
    API.clone()
}

pub fn is_available() -> bool {
    api().is_some()
}

pub fn device_name() -> String {
    api().map_or_else(
        || "Intel XPU (OpenCL)".to_string(),
        |a| a.device_name().to_string(),
    )
}

pub fn supports_op(op: &Op, dtype: DType) -> bool {
    crate::graph::portable::supports_op(op, dtype)
}

pub fn bandwidth_gbps() -> Option<f64> {
    api().and_then(|a| crate::graph::portable::bandwidth_gbps(&*a))
}
