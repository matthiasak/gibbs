//! In-memory ANE programs: MIL text plus one `weight.bin`, compiled and loaded through the private
//! `AppleNeuralEngine.framework`, evaluated over reusable IOSurface inputs and outputs.

// objc 0.2's `msg_send!` expands a `cfg(feature = "cargo-clippy")` check.
#![allow(unexpected_cfgs)]

use objc::runtime::{Class, Object, BOOL, NO};
use objc::{class, msg_send, sel, sel_impl};
use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;

pub(crate) type Id = *mut Object;

/// Programs compiled per process before new ANE segments are refused (the framework leaks
/// compiler resources and fails at roughly 119 compiles).
pub const ANE_MAX_PROGRAMS: usize = 100;

static COMPILED: AtomicUsize = AtomicUsize::new(0);

/// Programs this process compiled (cache hits through `compiledModelExists` are not counted).
pub fn programs_compiled() -> usize {
    COMPILED.load(Ordering::Relaxed)
}

/// `QOS_CLASS_USER_INTERACTIVE`, the QoS every compile, load and eval runs at.
const QOS: u32 = 21;

const FRAMEWORK: &str =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceCreate(properties: Id) -> *mut c_void;
    fn IOSurfaceLock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceUnlock(surface: *mut c_void, options: u32, seed: *mut u32) -> i32;
    fn IOSurfaceGetBaseAddress(surface: *mut c_void) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: *mut c_void) -> usize;
}

#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFRelease(cf: *const c_void);
}

#[link(name = "Foundation", kind = "framework")]
extern "C" {
    fn NSTemporaryDirectory() -> Id;
}

/// Loads the framework once; errors name what is missing.
pub(crate) fn framework() -> Result<(), String> {
    static LOADED: LazyLock<Result<(), String>> = LazyLock::new(|| {
        let path = std::ffi::CString::new(FRAMEWORK).expect("no NUL");
        // SAFETY: dlopen of a system framework path; the handle stays loaded for the process.
        if unsafe { libc::dlopen(path.as_ptr(), libc::RTLD_NOW) }.is_null() {
            return Err(format!("dlopen {FRAMEWORK} failed"));
        }
        for name in [
            "_ANEInMemoryModelDescriptor",
            "_ANEInMemoryModel",
            "_ANERequest",
            "_ANEIOSurfaceObject",
            "_ANEDeviceInfo",
            "_ANEClient",
        ] {
            class(name)?;
        }
        Ok(())
    });
    LOADED.clone()
}

pub(crate) fn class(name: &str) -> Result<&'static Class, String> {
    Class::get(name).ok_or_else(|| format!("AppleNeuralEngine: class {name} not found"))
}

/// Autoreleased `NSString`.
pub(crate) unsafe fn nsstring(s: &str) -> Id {
    let ns: Id = msg_send![class!(NSString), alloc];
    let ns: Id =
        msg_send![ns, initWithBytes: s.as_ptr() as *const c_void length: s.len() encoding: 4usize];
    msg_send![ns, autorelease]
}

pub(crate) unsafe fn string(ns: Id) -> String {
    if ns.is_null() {
        return String::new();
    }
    let p: *const std::ffi::c_char = msg_send![ns, UTF8String];
    if p.is_null() {
        return String::new();
    }
    std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
}

unsafe fn number(v: u64) -> Id {
    msg_send![class!(NSNumber), numberWithUnsignedLongLong: v]
}

unsafe fn dict(pairs: &[(Id, Id)]) -> Id {
    let d: Id = msg_send![class!(NSMutableDictionary), dictionary];
    for &(k, v) in pairs {
        let () = msg_send![d, setObject: v forKey: k];
    }
    d
}

unsafe fn array(items: &[Id]) -> Id {
    let a: Id = msg_send![class!(NSMutableArray), array];
    for &v in items {
        let () = msg_send![a, addObject: v];
    }
    a
}

unsafe fn describe(err: Id) -> String {
    if err.is_null() {
        return "unknown error".into();
    }
    let d: Id = msg_send![err, description];
    string(d)
}

/// A shared-memory buffer the ANE reads or writes (`[1, C, 1, S]` tensors, channel-major).
pub struct Surface {
    raw: *mut c_void,
}

// IOSurfaces are reference-counted kernel objects usable from any thread.
unsafe impl Send for Surface {}

impl Surface {
    pub fn new(bytes: usize) -> Result<Self, String> {
        let raw = objc::rc::autoreleasepool(|| unsafe {
            let n = number(bytes as u64);
            let props = dict(&[
                (nsstring("IOSurfaceWidth"), n),
                (nsstring("IOSurfaceHeight"), number(1)),
                (nsstring("IOSurfaceBytesPerElement"), number(1)),
                (nsstring("IOSurfaceBytesPerRow"), n),
                (nsstring("IOSurfaceAllocSize"), n),
                (nsstring("IOSurfacePixelFormat"), number(0)),
            ]);
            IOSurfaceCreate(props)
        });
        if raw.is_null() {
            return Err(format!("IOSurfaceCreate({bytes} bytes) failed"));
        }
        Ok(Self { raw })
    }

    pub fn alloc_size(&self) -> usize {
        unsafe { IOSurfaceGetAllocSize(self.raw) }
    }

    /// Base address (page-aligned; Metal can wrap it without copying).
    pub fn base(&self) -> *mut u8 {
        unsafe { IOSurfaceGetBaseAddress(self.raw) as *mut u8 }
    }

    /// Runs `f` over the locked surface bytes for writing.
    pub fn write<R>(&self, f: impl FnOnce(&mut [u8]) -> R) -> R {
        unsafe {
            IOSurfaceLock(self.raw, 0, std::ptr::null_mut());
            let r = f(std::slice::from_raw_parts_mut(
                self.base(),
                self.alloc_size(),
            ));
            IOSurfaceUnlock(self.raw, 0, std::ptr::null_mut());
            r
        }
    }

    /// Runs `f` over the surface bytes locked read-only.
    pub fn read<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        const READ_ONLY: u32 = 1;
        unsafe {
            IOSurfaceLock(self.raw, READ_ONLY, std::ptr::null_mut());
            let r = f(std::slice::from_raw_parts(self.base(), self.alloc_size()));
            IOSurfaceUnlock(self.raw, READ_ONLY, std::ptr::null_mut());
            r
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { CFRelease(self.raw) }
    }
}

/// How an eval is submitted; [`super::probe`] picks the first that works on this host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EvalEntry {
    /// `_ANEClient -evaluateRealTimeWithModel:options:request:error:`.
    RealTime,
    /// `_ANEClient -doEvaluateDirectWithModel:options:request:qos:error:`.
    Direct,
    /// `_ANEInMemoryModel -evaluateWithQoS:options:request:error:`.
    InMemory,
}

/// A compiled and loaded ANE program with one reusable request over its surfaces.
pub struct AneProgram {
    model: Id,
    request: Id,
    inputs: Vec<Surface>,
    outputs: Vec<Surface>,
    dir: PathBuf,
}

// The model and request are thread-safe reference-counted objects; `eval` takes `&mut self`.
unsafe impl Send for AneProgram {}

impl AneProgram {
    /// Compiles (unless the framework's cache already holds it) and loads `mil` with `weights`
    /// (`weight.bin`; empty for weight-free programs), then binds `n_in` input and `n_out` output
    /// surfaces of `in_bytes` / `out_bytes` each (in MIL argument order).
    pub fn compile(
        mil: &str,
        weights: &[u8],
        n_in: usize,
        in_bytes: usize,
        n_out: usize,
        out_bytes: usize,
    ) -> Result<Self, String> {
        framework()?;
        let inputs = (0..n_in)
            .map(|_| Surface::new(in_bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = (0..n_out)
            .map(|_| Surface::new(out_bytes))
            .collect::<Result<Vec<_>, _>>()?;
        objc::rc::autoreleasepool(|| unsafe {
            let mil_data: Id = msg_send![class!(NSData), dataWithBytes: mil.as_ptr() as *const c_void length: mil.len()];
            let wdict = if weights.is_empty() {
                dict(&[])
            } else {
                let data: Id = msg_send![class!(NSData), dataWithBytes: weights.as_ptr() as *const c_void length: weights.len()];
                dict(&[(
                    nsstring("@model_path/weights/weight.bin"),
                    dict(&[(nsstring("offset"), number(0)), (nsstring("data"), data)]),
                )])
            };
            let desc: Id = msg_send![class("_ANEInMemoryModelDescriptor")?, modelWithMILText: mil_data weights: wdict optionsPlist: std::ptr::null_mut::<Object>()];
            if desc.is_null() {
                return Err("ANE: modelWithMILText:weights:optionsPlist: returned nil".to_string());
            }
            let model: Id =
                msg_send![class("_ANEInMemoryModel")?, inMemoryModelWithDescriptor: desc];
            if model.is_null() {
                return Err("ANE: inMemoryModelWithDescriptor: returned nil".to_string());
            }
            let hex: Id = msg_send![model, hexStringIdentifier];
            let dir = PathBuf::from(string(NSTemporaryDirectory())).join(string(hex));
            let io = |e: std::io::Error| format!("ANE model dir {}: {e}", dir.display());
            std::fs::create_dir_all(dir.join("weights")).map_err(io)?;
            std::fs::write(dir.join("model.mil"), mil).map_err(io)?;
            if !weights.is_empty() {
                std::fs::write(dir.join("weights/weight.bin"), weights).map_err(io)?;
            }
            let opts = dict(&[]);
            let mut err: Id = std::ptr::null_mut();
            let cached: BOOL = msg_send![model, compiledModelExists];
            if cached == NO {
                if COMPILED.load(Ordering::Relaxed) >= ANE_MAX_PROGRAMS {
                    let _ = std::fs::remove_dir_all(&dir);
                    return Err(format!(
                        "ANE program budget reached ({ANE_MAX_PROGRAMS} compiles)"
                    ));
                }
                let ok: BOOL = msg_send![model, compileWithQoS: QOS options: opts error: &mut err];
                COMPILED.fetch_add(1, Ordering::Relaxed);
                if ok == NO {
                    let _ = std::fs::remove_dir_all(&dir);
                    return Err(format!("ANE compile: {}", describe(err)));
                }
            }
            let mut ok: BOOL = msg_send![model, loadWithQoS: QOS options: opts error: &mut err];
            if ok == NO {
                // A just-unloaded program's slot can take a moment to free.
                std::thread::sleep(std::time::Duration::from_millis(100));
                ok = msg_send![model, loadWithQoS: QOS options: opts error: &mut err];
            }
            if ok == NO {
                let _ = std::fs::remove_dir_all(&dir);
                return Err(format!("ANE load: {}", describe(err)));
            }
            let io_class = class("_ANEIOSurfaceObject")?;
            let wrap = |s: &Surface| -> Id { msg_send![io_class, objectWithIOSurface: s.raw] };
            let ins: Vec<Id> = inputs.iter().map(wrap).collect();
            let outs: Vec<Id> = outputs.iter().map(wrap).collect();
            let in_idx: Vec<Id> = (0..n_in).map(|i| number(i as u64)).collect();
            let out_idx: Vec<Id> = (0..n_out).map(|i| number(i as u64)).collect();
            let nil = std::ptr::null_mut::<Object>();
            let request: Id = msg_send![class("_ANERequest")?,
                requestWithInputs: array(&ins)
                inputIndices: array(&in_idx)
                outputs: array(&outs)
                outputIndices: array(&out_idx)
                weightsBuffer: nil
                perfStats: nil
                procedureIndex: number(0)];
            if request.is_null() {
                let _: BOOL = msg_send![model, unloadWithQoS: QOS error: &mut err];
                let _ = std::fs::remove_dir_all(&dir);
                return Err("ANE: _ANERequest construction returned nil".to_string());
            }
            let _: Id = msg_send![model, retain];
            let _: Id = msg_send![request, retain];
            Ok(Self {
                model,
                request,
                inputs,
                outputs,
                dir,
            })
        })
    }

    pub fn input(&self, i: usize) -> &Surface {
        &self.inputs[i]
    }

    pub fn output(&self, i: usize) -> &Surface {
        &self.outputs[i]
    }

    /// Evaluates the program once, synchronously, through the entry point [`super::probe`] chose.
    pub fn eval(&mut self) -> Result<(), String> {
        self.eval_with(super::probe()?.eval)
    }

    /// Evaluates the program once through `entry`, synchronously.
    pub(crate) fn eval_with(&mut self, entry: EvalEntry) -> Result<(), String> {
        objc::rc::autoreleasepool(|| unsafe {
            let opts = dict(&[]);
            let mut err: Id = std::ptr::null_mut();
            let ok: BOOL = match entry {
                EvalEntry::InMemory => {
                    msg_send![self.model, evaluateWithQoS: QOS options: opts request: self.request error: &mut err]
                }
                EvalEntry::RealTime | EvalEntry::Direct => {
                    let client: Id = msg_send![class("_ANEClient")?, sharedConnection];
                    let inner: Id = msg_send![self.model, model];
                    if client.is_null() || inner.is_null() {
                        return Err("ANE: no _ANEClient connection or _ANEModel".to_string());
                    }
                    if entry == EvalEntry::RealTime {
                        msg_send![client, evaluateRealTimeWithModel: inner options: opts request: self.request error: &mut err]
                    } else {
                        msg_send![client, doEvaluateDirectWithModel: inner options: opts request: self.request qos: QOS error: &mut err]
                    }
                }
            };
            if ok == NO {
                return Err(format!("ANE eval ({entry:?}): {}", describe(err)));
            }
            Ok(())
        })
    }
}

impl Drop for AneProgram {
    fn drop(&mut self) {
        objc::rc::autoreleasepool(|| unsafe {
            let mut err: Id = std::ptr::null_mut();
            let _: BOOL = msg_send![self.model, unloadWithQoS: QOS error: &mut err];
            let () = msg_send![self.request, release];
            let () = msg_send![self.model, release];
        });
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
