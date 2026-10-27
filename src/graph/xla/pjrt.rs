//! PJRT C API client (`xla/pjrt/c/pjrt_c_api.h`; argument layouts as of API 0.47, which later
//! versions only extend). A function is called only when its `PJRT_Api` field offset lies below
//! the plugin's `struct_size`, so an older plugin reports a missing entry point instead of
//! jumping through garbage.

use crate::graph::plan::DynLib;
use std::ffi::c_void;

type Raw = *mut c_void;
type Fn1 = unsafe extern "C" fn(Raw) -> Raw;

/// `PJRT_Api` function-field indices (declaration order in `pjrt_c_api.h`); field `i` sits at
/// byte `40 + 8 * i` after `struct_size`, `extension_start` and the 24-byte `PJRT_Api_Version`.
mod field {
    pub const ERROR_DESTROY: usize = 0;
    pub const ERROR_MESSAGE: usize = 1;
    pub const PLUGIN_INITIALIZE: usize = 3;
    pub const EVENT_DESTROY: usize = 5;
    pub const EVENT_AWAIT: usize = 8;
    pub const CLIENT_CREATE: usize = 10;
    pub const CLIENT_DESTROY: usize = 11;
    pub const CLIENT_PLATFORM_NAME: usize = 12;
    pub const CLIENT_ADDRESSABLE_DEVICES: usize = 16;
    pub const CLIENT_COMPILE: usize = 20;
    pub const CLIENT_BUFFER_FROM_HOST: usize = 22;
    pub const LOADED_EXECUTABLE_DESTROY: usize = 50;
    pub const LOADED_EXECUTABLE_EXECUTE: usize = 55;
    pub const BUFFER_DESTROY: usize = 58;
    pub const BUFFER_TO_HOST: usize = 70;
}

macro_rules! args {
    ($name:ident { $($f:ident : $t:ty),* $(,)? }) => {
        #[repr(C)]
        struct $name {
            struct_size: usize,
            extension_start: Raw,
            $($f: $t),*
        }
        impl $name {
            fn new() -> Self {
                // SAFETY: every field is a pointer, integer or enum for which zero is valid.
                let mut a: Self = unsafe { std::mem::zeroed() };
                a.struct_size = std::mem::size_of::<Self>();
                a
            }
        }
    };
}

args!(ErrorDestroy { error: Raw });
args!(ErrorMessage { error: Raw, message: *const u8, message_size: usize });
args!(PluginInitialize {});
args!(EventAwait { event: Raw });
args!(ClientCreate {
    create_options: Raw,
    num_options: usize,
    kv_get_callback: Raw,
    kv_get_user_arg: Raw,
    kv_put_callback: Raw,
    kv_put_user_arg: Raw,
    client: Raw,
});
args!(ClientRef { client: Raw });
args!(ClientPlatformName { client: Raw, platform_name: *const u8, platform_name_size: usize });
args!(ClientDevices { client: Raw, devices: *const Raw, num_devices: usize });
args!(Program { code: *const u8, code_size: usize, format: *const u8, format_size: usize });
args!(ClientCompile {
    client: Raw,
    program: *const Program,
    compile_options: *const u8,
    compile_options_size: usize,
    executable: Raw,
});
args!(BufferFromHost {
    client: Raw,
    data: *const c_void,
    ty: i32,
    dims: *const i64,
    num_dims: usize,
    byte_strides: *const i64,
    num_byte_strides: usize,
    host_buffer_semantics: i32,
    device: Raw,
    memory: Raw,
    device_layout: Raw,
    done_with_host_buffer: Raw,
    buffer: Raw,
});
args!(ExecuteOptions {
    send_callbacks: Raw,
    recv_callbacks: Raw,
    num_send_ops: usize,
    num_recv_ops: usize,
    launch_id: i32,
    non_donatable_input_indices: *const i64,
    num_non_donatable_input_indices: usize,
});
args!(Execute {
    executable: Raw,
    options: *mut ExecuteOptions,
    argument_lists: *const *const Raw,
    num_devices: usize,
    num_args: usize,
    output_lists: *const *mut Raw,
    device_complete_events: *mut Raw,
    execute_device: Raw,
});
args!(ObjRef { obj: Raw });
args!(BufferToHost { src: Raw, host_layout: Raw, dst: *mut c_void, dst_size: usize, event: Raw });

/// `PJRT_Buffer_Type` element types this client moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElemType {
    S32 = 4,
    F32 = 11,
    BF16 = 13,
}

/// `CompileOptionsProto { executable_build_options (3): { num_replicas (4): 1, num_partitions (5): 1 } }`
/// (`xla/pjrt/compile_options.proto`), varints written by hand.
const COMPILE_OPTIONS: [u8; 6] = [0x1a, 0x04, 0x20, 0x01, 0x28, 0x01];

/// The loaded plugin's function table.
#[derive(Clone, Copy)]
struct Api {
    base: *const u8,
    size: usize,
}

// SAFETY: `PJRT_Api` is an immutable, process-wide table of thread-safe entry points.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    fn func(&self, index: usize, name: &str) -> Result<Fn1, String> {
        let off = 40 + 8 * index;
        if off + 8 > self.size {
            return Err(format!(
                "PJRT plugin has no PJRT_{name} (PJRT_Api struct_size {})",
                self.size
            ));
        }
        // SAFETY: `off` lies inside the table the plugin reports.
        let p = unsafe { *(self.base.add(off) as *const Raw) };
        if p.is_null() {
            return Err(format!("PJRT plugin leaves PJRT_{name} unset"));
        }
        // SAFETY: every PJRT entry point is `PJRT_Error* fn(Args*)`.
        Ok(unsafe { std::mem::transmute::<Raw, Fn1>(p) })
    }

    /// Calls entry `index` with `args`; a returned `PJRT_Error` becomes `Err` naming the call.
    fn call<A>(&self, index: usize, name: &str, args: &mut A) -> Result<(), String> {
        let f = self.func(index, name)?;
        // SAFETY: `args` is the `repr(C)` argument struct of entry `index`.
        let err = unsafe { f(args as *mut A as Raw) };
        if err.is_null() {
            return Ok(());
        }
        let mut m = ErrorMessage::new();
        m.error = err;
        let text = match self.func(field::ERROR_MESSAGE, "Error_Message") {
            // SAFETY: as above; the message stays valid until the error is destroyed.
            Ok(f) => unsafe {
                f(&mut m as *mut ErrorMessage as Raw);
                String::from_utf8_lossy(std::slice::from_raw_parts(m.message, m.message_size))
                    .into_owned()
            },
            Err(e) => e,
        };
        if let Ok(f) = self.func(field::ERROR_DESTROY, "Error_Destroy") {
            let mut d = ErrorDestroy::new();
            d.error = err;
            // SAFETY: destroys the error returned above exactly once.
            unsafe { f(&mut d as *mut ErrorDestroy as Raw) };
        }
        Err(format!("PJRT_{name}: {text}"))
    }

    /// Awaits and destroys `event` (null: nothing to wait for).
    fn wait(&self, event: Raw) -> Result<(), String> {
        if event.is_null() {
            return Ok(());
        }
        let mut a = EventAwait::new();
        a.event = event;
        let r = self.call(field::EVENT_AWAIT, "Event_Await", &mut a);
        let mut d = ObjRef::new();
        d.obj = event;
        self.call(field::EVENT_DESTROY, "Event_Destroy", &mut d)?;
        r
    }

    fn destroy(&self, index: usize, name: &str, obj: Raw) {
        let mut d = ObjRef::new();
        d.obj = obj;
        if let Err(e) = self.call(index, name, &mut d) {
            eprintln!("warning: {e}");
        }
    }
}

/// A device buffer (destroyed on drop).
pub struct Buffer {
    raw: Raw,
    api: Api,
    /// Bytes on the host side of a download.
    pub bytes: usize,
}

// SAFETY: PJRT buffers may be used and destroyed from any thread.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.api
            .destroy(field::BUFFER_DESTROY, "Buffer_Destroy", self.raw);
    }
}

/// A compiled, loaded executable (destroyed on drop).
pub struct Executable {
    raw: Raw,
    api: Api,
    /// Result count (the module's `main` returns this many values).
    pub outputs: usize,
}

// SAFETY: as for `Buffer`.
unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Drop for Executable {
    fn drop(&mut self) {
        self.api.destroy(
            field::LOADED_EXECUTABLE_DESTROY,
            "LoadedExecutable_Destroy",
            self.raw,
        );
    }
}

/// One plugin, its client and the first addressable device.
pub struct Client {
    api: Api,
    client: Raw,
    device: Raw,
    /// `PJRT_Client_PlatformName` (`METAL`, `tpu`, `cuda`, ...).
    pub platform: String,
    /// `PJRT_Api.pjrt_api_version` (major, minor).
    pub version: (i32, i32),
    /// The plugin's `PJRT_Api.struct_size`.
    pub struct_size: usize,
    pub path: String,
    _lib: DynLib,
}

// SAFETY: a PJRT client is thread-safe; the device handle is owned by it.
unsafe impl Send for Client {}
unsafe impl Sync for Client {}

impl Drop for Client {
    fn drop(&mut self) {
        let mut a = ClientRef::new();
        a.client = self.client;
        if let Err(e) = self.api.call(field::CLIENT_DESTROY, "Client_Destroy", &mut a) {
            eprintln!("warning: {e}");
        }
    }
}

impl Client {
    /// Loads the plugin at `path`, initializes it and creates a client on its first device.
    pub fn open(path: &str) -> Result<Self, String> {
        let lib = DynLib::open_any(&[path]).ok_or_else(|| format!("cannot dlopen {path}"))?;
        let get = lib
            .sym_ptr(c"GetPjrtApi")
            .ok_or_else(|| format!("{path} has no GetPjrtApi"))?;
        // SAFETY: `GetPjrtApi` is `const PJRT_Api* GetPjrtApi(void)`.
        let base = unsafe {
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> *const u8>(get)()
        };
        if base.is_null() {
            return Err(format!("{path}: GetPjrtApi returned null"));
        }
        // SAFETY: the table starts with `size_t struct_size`, then the version struct at 16.
        let (size, version) = unsafe {
            (
                *(base as *const usize),
                (
                    *(base.add(32) as *const i32),
                    *(base.add(36) as *const i32),
                ),
            )
        };
        let api = Api { base, size };
        api.call(
            field::PLUGIN_INITIALIZE,
            "Plugin_Initialize",
            &mut PluginInitialize::new(),
        )?;
        let mut c = ClientCreate::new();
        api.call(field::CLIENT_CREATE, "Client_Create", &mut c)?;
        let mut client = Client {
            api,
            client: c.client,
            device: std::ptr::null_mut(),
            platform: String::new(),
            version,
            struct_size: size,
            path: path.to_string(),
            _lib: lib,
        };
        let mut n = ClientPlatformName::new();
        n.client = client.client;
        api.call(field::CLIENT_PLATFORM_NAME, "Client_PlatformName", &mut n)?;
        // SAFETY: the name is owned by the client and `platform_name_size` bytes long.
        client.platform = String::from_utf8_lossy(unsafe {
            std::slice::from_raw_parts(n.platform_name, n.platform_name_size)
        })
        .into_owned();
        let mut d = ClientDevices::new();
        d.client = client.client;
        api.call(
            field::CLIENT_ADDRESSABLE_DEVICES,
            "Client_AddressableDevices",
            &mut d,
        )?;
        if d.num_devices == 0 {
            return Err(format!("{path}: no addressable PJRT device"));
        }
        // SAFETY: `num_devices > 0` entries owned by the client.
        client.device = unsafe { *d.devices };
        Ok(client)
    }

    /// Compiles StableHLO MLIR text returning `outputs` results.
    pub fn compile(&self, mlir: &str, outputs: usize) -> Result<Executable, String> {
        // Plugins parse past `code_size` looking for a terminator; keep one there.
        let mut code = Vec::with_capacity(mlir.len() + 1);
        code.extend_from_slice(mlir.as_bytes());
        code.push(0);
        let mut p = Program::new();
        (p.code, p.code_size) = (code.as_ptr(), mlir.len());
        (p.format, p.format_size) = (b"mlir".as_ptr(), 4);
        let mut a = ClientCompile::new();
        a.client = self.client;
        a.program = &p;
        (a.compile_options, a.compile_options_size) =
            (COMPILE_OPTIONS.as_ptr(), COMPILE_OPTIONS.len());
        self.api
            .call(field::CLIENT_COMPILE, "Client_Compile", &mut a)?;
        Ok(Executable {
            raw: a.executable,
            api: self.api,
            outputs,
        })
    }

    /// Copies `data` (row-major `dims` of `ty`) to the device.
    pub fn upload(&self, data: &[u8], ty: ElemType, dims: &[usize]) -> Result<Buffer, String> {
        let dims: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let mut a = BufferFromHost::new();
        a.client = self.client;
        a.data = data.as_ptr() as *const c_void;
        a.ty = ty as i32;
        (a.dims, a.num_dims) = (dims.as_ptr(), dims.len());
        // kImmutableOnlyDuringCall: the plugin copies before returning.
        a.host_buffer_semantics = 0;
        a.device = self.device;
        self.api.call(
            field::CLIENT_BUFFER_FROM_HOST,
            "Client_BufferFromHostBuffer",
            &mut a,
        )?;
        let buf = Buffer {
            raw: a.buffer,
            api: self.api,
            bytes: data.len(),
        };
        self.api.wait(a.done_with_host_buffer)?;
        Ok(buf)
    }

    /// Runs `exe` on `args` and returns its results; `result_bytes[i]` is result `i`'s host size.
    /// Arguments the executable donates (`tf.aliasing_output`) are consumed by the plugin; their
    /// handles are still destroyed by the caller.
    pub fn execute(
        &self,
        exe: &Executable,
        args: &[&Buffer],
        result_bytes: &[usize],
    ) -> Result<Vec<Buffer>, String> {
        let raw_args: Vec<Raw> = args.iter().map(|b| b.raw).collect();
        let arg_list = [raw_args.as_ptr()];
        let mut outs: Vec<Raw> = vec![std::ptr::null_mut(); exe.outputs];
        let out_list = [outs.as_mut_ptr()];
        let mut opts = ExecuteOptions::new();
        let mut a = Execute::new();
        a.executable = exe.raw;
        a.options = &mut opts;
        a.argument_lists = arg_list.as_ptr();
        a.num_devices = 1;
        a.num_args = raw_args.len();
        a.output_lists = out_list.as_ptr();
        self.api
            .call(field::LOADED_EXECUTABLE_EXECUTE, "LoadedExecutable_Execute", &mut a)?;
        Ok(outs
            .into_iter()
            .zip(result_bytes)
            .map(|(raw, &bytes)| Buffer {
                raw,
                api: self.api,
                bytes,
            })
            .collect())
    }

    /// Copies `buf` to the host (`dst.len()` must be its byte size).
    pub fn download(&self, buf: &Buffer, dst: &mut [u8]) -> Result<(), String> {
        let mut a = BufferToHost::new();
        a.src = buf.raw;
        (a.dst, a.dst_size) = (dst.as_mut_ptr() as *mut c_void, dst.len());
        self.api
            .call(field::BUFFER_TO_HOST, "Buffer_ToHostBuffer", &mut a)?;
        self.api.wait(a.event)
    }
}
