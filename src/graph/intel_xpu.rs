//! Intel XPU backend: an OpenCL 1.2 [`GpuApi`] (`libOpenCL.so.1`, or `OpenCL.framework` on
//! macOS, loaded at runtime through [`DynLib`]) running the portable kernels.
//!
//! Prefers oneAPI Level Zero (`libze_loader` + `libocloc`), falling back to OpenCL 1.2 (trying
//! `Intel` platforms first, then any GPU platform unless `GIBBS_XPU_ANY_OPENCL=0`).
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
const CL_PROGRAM_BINARY_SIZES: u32 = 0x1165;
const CL_PROGRAM_BINARIES: u32 = 0x1166;
const CL_FALSE: u32 = 0;
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
    create_program_with_binary: unsafe extern "C" fn(
        Handle,
        u32,
        *const Handle,
        *const usize,
        *const *const u8,
        *mut ClInt,
        *mut ClInt,
    ) -> Handle,
    release_program: unsafe extern "C" fn(Handle) -> ClInt,
    get_program_info: unsafe extern "C" fn(Handle, u32, usize, *mut c_void, *mut usize) -> ClInt,
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
            create_program_with_binary: sym!("clCreateProgramWithBinary"),
            release_program: sym!("clReleaseProgram"),
            get_program_info: sym!("clGetProgramInfo"),
            build_program: sym!("clBuildProgram"),
            get_program_build_info: sym!("clGetProgramBuildInfo"),
            create_kernel: sym!("clCreateKernel"),
            set_kernel_arg: sym!("clSetKernelArg"),
            enqueue_nd_range_kernel: sym!("clEnqueueNDRangeKernel"),
            finish: sym!("clFinish"),
        };
        let any = !matches!(
            std::env::var("GIBBS_XPU_ANY_OPENCL").as_deref(),
            Ok("0" | "off" | "false")
        );
        let mut n = 0u32;
        // SAFETY: count query, then a buffer of exactly `n` platform ids.
        check("clGetPlatformIDs", unsafe {
            (f.get_platform_ids)(0, std::ptr::null_mut(), &mut n)
        })?;
        let mut plats = vec![std::ptr::null_mut(); n as usize];
        check("clGetPlatformIDs", unsafe {
            (f.get_platform_ids)(n, plats.as_mut_ptr(), &mut n)
        })?;
        for &intel_only in if any { &[true, false][..] } else { &[true][..] } {
            for &p in &plats {
                let pname = info_string(f.get_platform_info, p, CL_PLATFORM_NAME);
                if intel_only && !pname.contains("Intel") {
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

    fn upload_async(&self, buf: Buf, offset: usize, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: in-order OpenCL queue orders non-blocking writes before subsequent kernels,
        // and `PortableSession` keeps borrowed weight slices alive until `sync()` / session drop.
        check("clEnqueueWriteBuffer(async)", unsafe {
            (self.f.enqueue_write_buffer)(
                self.queue,
                buf.0 as Handle,
                CL_FALSE,
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

    fn compile_cached(&self, src: &str, file: &std::path::Path) -> Result<Module, String> {
        let opts = CString::new("-cl-std=CL1.2").expect("no NUL");
        if let Ok(bin) = std::fs::read(file) {
            if !bin.is_empty() {
                let len = bin.len();
                let ptr = bin.as_ptr();
                let (mut bstat, mut err) = (0, 0);
                // SAFETY: one device and one binary slice of `len` bytes.
                let prog = unsafe {
                    (self.f.create_program_with_binary)(
                        self.context,
                        1,
                        &self.device,
                        &len,
                        &ptr,
                        &mut bstat,
                        &mut err,
                    )
                };
                if err == 0 && bstat == 0 && !prog.is_null() {
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
                    if code == 0 {
                        return Ok(Module(prog as u64));
                    }
                    unsafe { (self.f.release_program)(prog) };
                } else if !prog.is_null() {
                    unsafe { (self.f.release_program)(prog) };
                }
                let _ = std::fs::remove_file(file);
            }
        }
        let m = self.compile(src)?;
        let prog = m.0 as Handle;
        let mut bsize = 0usize;
        // SAFETY: queries the one device's binary size, then fetches into an array of 1 pointer.
        let scode = unsafe {
            (self.f.get_program_info)(
                prog,
                CL_PROGRAM_BINARY_SIZES,
                std::mem::size_of::<usize>(),
                (&mut bsize as *mut usize).cast(),
                std::ptr::null_mut(),
            )
        };
        if scode == 0 && bsize > 0 {
            let mut bin = vec![0u8; bsize];
            let mut bptr: *mut u8 = bin.as_mut_ptr();
            let bcode = unsafe {
                (self.f.get_program_info)(
                    prog,
                    CL_PROGRAM_BINARIES,
                    std::mem::size_of::<*mut u8>(),
                    (&mut bptr as *mut *mut u8).cast(),
                    std::ptr::null_mut(),
                )
            };
            if bcode == 0 {
                let _ = crate::graph::portable::cached_bytes(file, || Ok(bin));
            }
        }
        Ok(m)
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

// ---- oneAPI Level Zero (`libze_loader` + `libocloc` SPIR-V) with OpenCL fallback ----

#[repr(C)]
struct ZeDeviceProperties {
    stype: u32,
    pnext: *mut c_void,
    dtype: u32,
    vendor_id: u32,
    device_id: u32,
    flags: u32,
    subdevice_id: u32,
    core_clock_rate: u32,
    max_mem_alloc_size: u64,
    max_hardware_contexts: u32,
    max_command_queue_priority: u32,
    num_threads_per_eu: u32,
    physical_eu_simd_width: u32,
    num_eus_per_sub_slice: u32,
    num_sub_slices_per_slice: u32,
    num_slices: u32,
    timer_resolution: u64,
    timestamp_valid_bits: u32,
    kernel_timestamp_valid_bits: u32,
    uuid: [u8; 16],
    name: [u8; 256],
}

#[repr(C)]
struct ZeContextDesc {
    stype: u32,
    pnext: *const c_void,
    flags: u32,
}

#[repr(C)]
struct ZeCommandQueueDesc {
    stype: u32,
    pnext: *const c_void,
    ordinal: u32,
    index: u32,
    flags: u32,
    mode: u32,
    priority: u32,
}

#[repr(C)]
struct ZeDeviceMemAllocDesc {
    stype: u32,
    pnext: *const c_void,
    flags: u32,
    ordinal: u32,
}

#[repr(C)]
struct ZeModuleDesc {
    stype: u32,
    pnext: *const c_void,
    format: u32,
    input_size: usize,
    p_input_module: *const u8,
    p_build_flags: *const c_char,
    p_constants: *const c_void,
}

#[repr(C)]
struct ZeKernelDesc {
    stype: u32,
    pnext: *const c_void,
    flags: u32,
    p_kernel_name: *const c_char,
}

#[repr(C)]
struct ZeGroupCount {
    x: u32,
    y: u32,
    z: u32,
}

#[allow(clippy::type_complexity)]
struct ZeFns {
    init: unsafe extern "C" fn(u32) -> i32,
    driver_get: unsafe extern "C" fn(*mut u32, *mut Handle) -> i32,
    device_get: unsafe extern "C" fn(Handle, *mut u32, *mut Handle) -> i32,
    device_get_properties: unsafe extern "C" fn(Handle, *mut ZeDeviceProperties) -> i32,
    context_create: unsafe extern "C" fn(Handle, *const ZeContextDesc, *mut Handle) -> i32,
    context_destroy: unsafe extern "C" fn(Handle) -> i32,
    cmd_list_create_immediate:
        unsafe extern "C" fn(Handle, Handle, *const ZeCommandQueueDesc, *mut Handle) -> i32,
    cmd_list_destroy: unsafe extern "C" fn(Handle) -> i32,
    cmd_list_host_sync: unsafe extern "C" fn(Handle, u64) -> i32,
    cmd_list_append_barrier: unsafe extern "C" fn(Handle, Handle, u32, *const Handle) -> i32,
    cmd_list_append_mem_copy: unsafe extern "C" fn(
        Handle,
        *mut c_void,
        *const c_void,
        usize,
        Handle,
        u32,
        *const Handle,
    ) -> i32,
    cmd_list_append_mem_fill: unsafe extern "C" fn(
        Handle,
        *mut c_void,
        *const c_void,
        usize,
        usize,
        Handle,
        u32,
        *const Handle,
    ) -> i32,
    mem_alloc_device: unsafe extern "C" fn(
        Handle,
        *const ZeDeviceMemAllocDesc,
        usize,
        usize,
        Handle,
        *mut *mut c_void,
    ) -> i32,
    mem_free: unsafe extern "C" fn(Handle, *mut c_void) -> i32,
    module_create:
        unsafe extern "C" fn(Handle, Handle, *const ZeModuleDesc, *mut Handle, *mut Handle) -> i32,
    kernel_create: unsafe extern "C" fn(Handle, *const ZeKernelDesc, *mut Handle) -> i32,
    kernel_set_group_size: unsafe extern "C" fn(Handle, u32, u32, u32) -> i32,
    kernel_set_arg_value: unsafe extern "C" fn(Handle, u32, usize, *const c_void) -> i32,
    cmd_list_append_launch_kernel: unsafe extern "C" fn(
        Handle,
        Handle,
        *const ZeGroupCount,
        Handle,
        u32,
        *const Handle,
    ) -> i32,
}

#[allow(clippy::type_complexity)]
struct OclocFns {
    invoke: unsafe extern "C" fn(
        u32,
        *const *const c_char,
        u32,
        *const *const u8,
        *const u64,
        *const *const c_char,
        u32,
        *const *const u8,
        *const u64,
        *const *const c_char,
        *mut u32,
        *mut *mut *mut u8,
        *mut *mut u64,
        *mut *mut *mut c_char,
    ) -> i32,
    free_output: unsafe extern "C" fn(
        *mut u32,
        *mut *mut *mut u8,
        *mut *mut u64,
        *mut *mut *mut c_char,
    ) -> i32,
}

struct LevelZeroApi {
    _ze_lib: DynLib,
    _ocloc_lib: DynLib,
    f: ZeFns,
    ocloc: OclocFns,
    context: Handle,
    device: Handle,
    cmd_list: Handle,
    name: String,
    launch: Mutex<()>,
}

// SAFETY: Level Zero calls are serialized on `launch` / the in-order immediate command list.
unsafe impl Send for LevelZeroApi {}
unsafe impl Sync for LevelZeroApi {}

fn ze_check(call: &str, code: i32) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{call}: Level Zero error {code:#x}"))
    }
}

impl LevelZeroApi {
    fn open() -> Result<Self, String> {
        let ze_lib = DynLib::open_any(&["libze_loader.so.1", "libze_loader.so", "ze_loader.dll"])
            .ok_or("libze_loader not found")?;
        let ocloc_lib =
            DynLib::open_any(&["libocloc.so", "ocloc64.dll"]).ok_or("libocloc not found")?;
        macro_rules! zsym {
            ($n:literal) => {
                unsafe {
                    std::mem::transmute(
                        ze_lib
                            .sym_ptr(
                                std::ffi::CStr::from_bytes_with_nul(concat!($n, "\0").as_bytes())
                                    .expect("literal"),
                            )
                            .ok_or(concat!("missing Level Zero symbol ", $n))?,
                    )
                }
            };
        }
        let f = ZeFns {
            init: zsym!("zeInit"),
            driver_get: zsym!("zeDriverGet"),
            device_get: zsym!("zeDeviceGet"),
            device_get_properties: zsym!("zeDeviceGetProperties"),
            context_create: zsym!("zeContextCreate"),
            context_destroy: zsym!("zeContextDestroy"),
            cmd_list_create_immediate: zsym!("zeCommandListCreateImmediate"),
            cmd_list_destroy: zsym!("zeCommandListDestroy"),
            cmd_list_host_sync: zsym!("zeCommandListHostSynchronize"),
            cmd_list_append_barrier: zsym!("zeCommandListAppendBarrier"),
            cmd_list_append_mem_copy: zsym!("zeCommandListAppendMemoryCopy"),
            cmd_list_append_mem_fill: zsym!("zeCommandListAppendMemoryFill"),
            mem_alloc_device: zsym!("zeMemAllocDevice"),
            mem_free: zsym!("zeMemFree"),
            module_create: zsym!("zeModuleCreate"),
            kernel_create: zsym!("zeKernelCreate"),
            kernel_set_group_size: zsym!("zeKernelSetGroupSize"),
            kernel_set_arg_value: zsym!("zeKernelSetArgumentValue"),
            cmd_list_append_launch_kernel: zsym!("zeCommandListAppendLaunchKernel"),
        };
        let ocloc = OclocFns {
            invoke: unsafe {
                std::mem::transmute(
                    ocloc_lib
                        .sym_ptr(c"oclocInvoke")
                        .ok_or("missing oclocInvoke")?,
                )
            },
            free_output: unsafe {
                std::mem::transmute(
                    ocloc_lib
                        .sym_ptr(c"oclocFreeOutput")
                        .ok_or("missing oclocFreeOutput")?,
                )
            },
        };
        // ZE_INIT_FLAG_GPU_ONLY = 1
        ze_check("zeInit", unsafe { (f.init)(1) })?;
        let mut ndrv = 0u32;
        ze_check("zeDriverGet(count)", unsafe {
            (f.driver_get)(&mut ndrv, std::ptr::null_mut())
        })?;
        if ndrv == 0 {
            return Err("no Level Zero drivers".into());
        }
        let mut drvs = vec![std::ptr::null_mut(); ndrv as usize];
        ze_check("zeDriverGet", unsafe {
            (f.driver_get)(&mut ndrv, drvs.as_mut_ptr())
        })?;
        for drv in drvs {
            let mut ndev = 0u32;
            if unsafe { (f.device_get)(drv, &mut ndev, std::ptr::null_mut()) } != 0 || ndev == 0 {
                continue;
            }
            let mut devs = vec![std::ptr::null_mut(); ndev as usize];
            if unsafe { (f.device_get)(drv, &mut ndev, devs.as_mut_ptr()) } != 0 {
                continue;
            }
            for dev in devs {
                // ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 3, ZE_DEVICE_TYPE_GPU = 1
                let mut props: ZeDeviceProperties = unsafe { std::mem::zeroed() };
                props.stype = 3;
                if unsafe { (f.device_get_properties)(dev, &mut props) } != 0 || props.dtype != 1 {
                    continue;
                }
                let ctx_desc = ZeContextDesc {
                    stype: 1, // ZE_STRUCTURE_TYPE_CONTEXT_DESC
                    pnext: std::ptr::null(),
                    flags: 0,
                };
                let mut context: Handle = std::ptr::null_mut();
                ze_check("zeContextCreate", unsafe {
                    (f.context_create)(drv, &ctx_desc, &mut context)
                })?;
                // ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 2, ZE_COMMAND_QUEUE_FLAG_IN_ORDER = 1,
                // ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2
                let q_desc = ZeCommandQueueDesc {
                    stype: 2,
                    pnext: std::ptr::null(),
                    ordinal: 0,
                    index: 0,
                    flags: 1,
                    mode: 2,
                    priority: 0,
                };
                let mut cmd_list: Handle = std::ptr::null_mut();
                if let Err(e) = ze_check("zeCommandListCreateImmediate", unsafe {
                    (f.cmd_list_create_immediate)(context, dev, &q_desc, &mut cmd_list)
                }) {
                    unsafe { (f.context_destroy)(context) };
                    return Err(e);
                }
                let nul = props
                    .name
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(props.name.len());
                let dname = String::from_utf8_lossy(&props.name[..nul])
                    .trim()
                    .to_string();
                return Ok(Self {
                    _ze_lib: ze_lib,
                    _ocloc_lib: ocloc_lib,
                    f,
                    ocloc,
                    context,
                    device: dev,
                    cmd_list,
                    name: format!("{dname} (Level Zero)"),
                    launch: Mutex::new(()),
                });
            }
        }
        Err("no Level Zero GPU device found".into())
    }

    fn compile_spirv(&self, src: &str) -> Result<Vec<u8>, String> {
        let args = [
            CString::new("ocloc").unwrap(),
            CString::new("compile").unwrap(),
            CString::new("-file").unwrap(),
            CString::new("gibbs.cl").unwrap(),
            CString::new("-spv_only").unwrap(),
            CString::new("-options").unwrap(),
            CString::new("-cl-std=CL1.2").unwrap(),
        ];
        let argv: Vec<*const c_char> = args.iter().map(|s| s.as_ptr()).collect();
        let fname = CString::new("gibbs.cl").unwrap();
        let src_data = src.as_ptr();
        let src_len = (src.len() + 1) as u64;
        let mut src_c = Vec::with_capacity(src.len() + 1);
        src_c.extend_from_slice(src.as_bytes());
        src_c.push(0);
        let data_ptr = src_c.as_ptr();
        let name_ptr = fname.as_ptr();
        let _ = src_data;
        let mut num_out = 0u32;
        let mut out_data: *mut *mut u8 = std::ptr::null_mut();
        let mut out_lens: *mut u64 = std::ptr::null_mut();
        let mut out_names: *mut *mut c_char = std::ptr::null_mut();
        let rc = unsafe {
            (self.ocloc.invoke)(
                argv.len() as u32,
                argv.as_ptr(),
                1,
                &data_ptr,
                &src_len,
                &name_ptr,
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                &mut num_out,
                &mut out_data,
                &mut out_lens,
                &mut out_names,
            )
        };
        let mut spv = None;
        if rc == 0 && num_out > 0 && !out_data.is_null() && !out_lens.is_null() {
            for i in 0..num_out as usize {
                let len = unsafe { *out_lens.add(i) } as usize;
                let ptr = unsafe { *out_data.add(i) };
                if len >= 4 && !ptr.is_null() {
                    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
                    // SPIR-V magic 0x07230203
                    if u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) == 0x0723_0203 {
                        spv = Some(slice.to_vec());
                        break;
                    }
                }
            }
        }
        if !out_data.is_null() {
            unsafe {
                (self.ocloc.free_output)(&mut num_out, &mut out_data, &mut out_lens, &mut out_names)
            };
        }
        spv.ok_or_else(|| format!("oclocInvoke failed (rc={rc})"))
    }

    fn load_spirv(&self, spv: &[u8]) -> Result<Module, String> {
        // ZE_STRUCTURE_TYPE_MODULE_DESC = 0x15, ZE_MODULE_FORMAT_IL_SPIRV = 0
        let desc = ZeModuleDesc {
            stype: 0x15,
            pnext: std::ptr::null(),
            format: 0,
            input_size: spv.len(),
            p_input_module: spv.as_ptr(),
            p_build_flags: std::ptr::null(),
            p_constants: std::ptr::null(),
        };
        let mut module: Handle = std::ptr::null_mut();
        ze_check("zeModuleCreate", unsafe {
            (self.f.module_create)(
                self.context,
                self.device,
                &desc,
                &mut module,
                std::ptr::null_mut(),
            )
        })?;
        Ok(Module(module as u64))
    }
}

impl Drop for LevelZeroApi {
    fn drop(&mut self) {
        unsafe {
            let _ = (self.f.cmd_list_host_sync)(self.cmd_list, u64::MAX);
            let _ = (self.f.cmd_list_destroy)(self.cmd_list);
            let _ = (self.f.context_destroy)(self.context);
        }
    }
}

impl GpuApi for LevelZeroApi {
    fn device_name(&self) -> &str {
        &self.name
    }

    fn dialect(&self) -> Dialect {
        Dialect::OpenCl
    }

    fn alloc(&self, bytes: usize) -> Result<Buf, String> {
        // ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x13
        let desc = ZeDeviceMemAllocDesc {
            stype: 0x13,
            pnext: std::ptr::null(),
            flags: 0,
            ordinal: 0,
        };
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let size = bytes.max(4);
        ze_check("zeMemAllocDevice", unsafe {
            (self.f.mem_alloc_device)(self.context, &desc, size, 64, self.device, &mut ptr)
        })?;
        let zero = 0u32;
        let _guard = self.launch.lock().unwrap_or_else(|e| e.into_inner());
        ze_check("zeCommandListAppendMemoryFill", unsafe {
            (self.f.cmd_list_append_mem_fill)(
                self.cmd_list,
                ptr,
                (&zero as *const u32).cast(),
                4,
                size,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        })?;
        Ok(Buf(ptr as u64))
    }

    fn free(&self, buf: Buf) {
        let _guard = self.launch.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            let _ = (self.f.cmd_list_host_sync)(self.cmd_list, u64::MAX);
            let _ = (self.f.mem_free)(self.context, buf.0 as *mut c_void);
        }
    }

    fn upload(&self, buf: Buf, offset: usize, data: &[u8]) -> Result<(), String> {
        self.upload_async(buf, offset, data)?;
        self.sync()
    }

    fn upload_async(&self, buf: Buf, offset: usize, data: &[u8]) -> Result<(), String> {
        if data.is_empty() {
            return Ok(());
        }
        let _guard = self.launch.lock().unwrap_or_else(|e| e.into_inner());
        let dst = (buf.0 as usize + offset) as *mut c_void;
        ze_check("zeCommandListAppendMemoryCopy(H2D)", unsafe {
            (self.f.cmd_list_append_mem_copy)(
                self.cmd_list,
                dst,
                data.as_ptr().cast(),
                data.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        })
    }

    fn download(&self, buf: Buf, offset: usize, out: &mut [u8]) -> Result<(), String> {
        if out.is_empty() {
            return Ok(());
        }
        let _guard = self.launch.lock().unwrap_or_else(|e| e.into_inner());
        let src = (buf.0 as usize + offset) as *const c_void;
        ze_check("zeCommandListAppendMemoryCopy(D2H)", unsafe {
            (self.f.cmd_list_append_mem_copy)(
                self.cmd_list,
                out.as_mut_ptr().cast(),
                src,
                out.len(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        })?;
        ze_check("zeCommandListHostSynchronize", unsafe {
            (self.f.cmd_list_host_sync)(self.cmd_list, u64::MAX)
        })
    }

    fn compile(&self, src: &str) -> Result<Module, String> {
        let spv = self.compile_spirv(src)?;
        self.load_spirv(&spv)
    }

    fn compile_cached(&self, src: &str, file: &std::path::Path) -> Result<Module, String> {
        let (spv, hit) = crate::graph::portable::cached_bytes(file, || self.compile_spirv(src))?;
        match self.load_spirv(&spv) {
            Ok(m) => Ok(m),
            Err(_) if hit => {
                let _ = std::fs::remove_file(file);
                let fresh = self.compile_spirv(src)?;
                let _ = crate::graph::portable::cached_bytes(file, || Ok(fresh.clone()));
                self.load_spirv(&fresh)
            }
            Err(e) => Err(e),
        }
    }

    fn kernel(&self, module: &Module, name: &str) -> Result<Kernel, String> {
        let cname = CString::new(name).map_err(|e| e.to_string())?;
        // ZE_STRUCTURE_TYPE_KERNEL_DESC = 0x17
        let desc = ZeKernelDesc {
            stype: 0x17,
            pnext: std::ptr::null(),
            flags: 0,
            p_kernel_name: cname.as_ptr(),
        };
        let mut k: Handle = std::ptr::null_mut();
        ze_check(&format!("zeKernelCreate({name})"), unsafe {
            (self.f.kernel_create)(module.0 as Handle, &desc, &mut k)
        })?;
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
        ze_check("zeKernelSetGroupSize", unsafe {
            (self.f.kernel_set_group_size)(kh, block[0], block[1], block[2])
        })?;
        for (i, a) in args.iter().enumerate() {
            let (size, ptr): (usize, *const c_void) = match a {
                Arg::Buf(b) => (8, (&b.0 as *const u64).cast()),
                Arg::U32(v) => (4, (v as *const u32).cast()),
                Arg::F32(v) => (4, (v as *const f32).cast()),
                Arg::U64(v) => (8, (v as *const u64).cast()),
            };
            ze_check(&format!("zeKernelSetArgumentValue({i})"), unsafe {
                (self.f.kernel_set_arg_value)(kh, i as u32, size, ptr)
            })?;
        }
        let groups = ZeGroupCount {
            x: grid[0],
            y: grid[1],
            z: grid[2],
        };
        ze_check("zeCommandListAppendLaunchKernel", unsafe {
            (self.f.cmd_list_append_launch_kernel)(
                self.cmd_list,
                kh,
                &groups,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        })?;
        let _ = self.f.cmd_list_append_barrier;
        Ok(())
    }

    fn sync(&self) -> Result<(), String> {
        let _guard = self.launch.lock().unwrap_or_else(|e| e.into_inner());
        ze_check("zeCommandListHostSynchronize", unsafe {
            (self.f.cmd_list_host_sync)(self.cmd_list, u64::MAX)
        })
    }
}

/// Tries oneAPI Level Zero (`libze_loader` + `libocloc`) first, falling back to OpenCL 1.2 when
/// Level Zero is unavailable or overridden (`GIBBS_XPU_OPENCL_ONLY=1`).
static API: LazyLock<Option<Arc<dyn GpuApi>>> = LazyLock::new(|| {
    if std::env::var("GIBBS_XPU_OPENCL_ONLY").as_deref() != Ok("1") {
        if let Ok(ze) = LevelZeroApi::open() {
            let api: Arc<dyn GpuApi> = Arc::new(ze);
            if crate::graph::portable::probe(&*api).is_ok() {
                return Some(api);
            }
        }
    }
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
