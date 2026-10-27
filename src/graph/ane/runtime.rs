//! In-memory ANE programs: MIL text plus one `weight.bin`, compiled and loaded through the private
//! `AppleNeuralEngine.framework`, evaluated over reusable IOSurface inputs and outputs.
//!
//! Every Objective-C object handed to a collection is null-checked first (inserting `nil` raises
//! an exception, which must not unwind through Rust), and every selector sent is checked once in
//! [`framework`], so an OS update that drops one fails with its name instead of an exception.

// objc 0.2's `msg_send!` expands a `cfg(feature = "cargo-clippy")` check.
#![allow(unexpected_cfgs)]

use objc::runtime::{Class, Object, Sel, BOOL, NO};
use objc::{class, msg_send, sel, sel_impl};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

pub(crate) type Id = *mut Object;

/// Compiles per process before new ANE programs are refused: the framework leaks compiler
/// resources on every compile attempt and fails at roughly 119.
pub const ANE_MAX_PROGRAMS: usize = 100;

static COMPILED: AtomicUsize = AtomicUsize::new(0);

/// Compile attempts this process made (programs the framework had cached are not counted).
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

/// `(class, class-method selectors, instance-method selectors)` this module sends.
const SELECTORS: [(&str, &[&str], &[&str]); 5] = [
    (
        "_ANEInMemoryModelDescriptor",
        &["modelWithMILText:weights:optionsPlist:"],
        &[],
    ),
    (
        "_ANEInMemoryModel",
        &["inMemoryModelWithDescriptor:"],
        &[
            "hexStringIdentifier",
            "compiledModelExists",
            "compileWithQoS:options:error:",
            "loadWithQoS:options:error:",
            "evaluateWithQoS:options:request:error:",
            "unloadWithQoS:error:",
            "model",
        ],
    ),
    (
        "_ANERequest",
        &["requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"],
        &[],
    ),
    ("_ANEIOSurfaceObject", &["objectWithIOSurface:"], &[]),
    (
        "_ANEDeviceInfo",
        &["hasANE", "numANECores", "aneArchitectureType"],
        &[],
    ),
];

/// Loads the framework once and checks every class and selector this module uses; errors name
/// what is missing.
pub(crate) fn framework() -> Result<(), String> {
    static LOADED: LazyLock<Result<(), String>> = LazyLock::new(|| {
        let path = std::ffi::CString::new(FRAMEWORK).expect("no NUL");
        // SAFETY: dlopen of a system framework path; the handle stays loaded for the process.
        // dlerror() is read right after on this thread.
        unsafe {
            if libc::dlopen(path.as_ptr(), libc::RTLD_NOW).is_null() {
                let e = libc::dlerror();
                let why = if e.is_null() {
                    "unknown error".into()
                } else {
                    std::ffi::CStr::from_ptr(e).to_string_lossy().into_owned()
                };
                return Err(format!("dlopen {FRAMEWORK}: {why}"));
            }
        }
        for (name, class_sels, inst_sels) in SELECTORS {
            let c = class(name)?;
            for s in class_sels {
                if c.metaclass().instance_method(Sel::register(s)).is_none() {
                    return Err(format!("AppleNeuralEngine: +[{name} {s}] not found"));
                }
            }
            for s in inst_sels {
                if c.instance_method(Sel::register(s)).is_none() {
                    return Err(format!("AppleNeuralEngine: -[{name} {s}] not found"));
                }
            }
        }
        Ok(())
    });
    LOADED.clone()
}

pub(crate) fn class(name: &str) -> Result<&'static Class, String> {
    Class::get(name).ok_or_else(|| format!("AppleNeuralEngine: class {name} not found"))
}

/// Whether instances of `class` answer `sel` (optional entry points).
pub(crate) fn responds(class: &Class, sel: &str) -> bool {
    class.instance_method(Sel::register(sel)).is_some()
}

fn nonnull(id: Id, what: &str) -> Result<Id, String> {
    if id.is_null() {
        Err(format!("ANE: {what} returned nil"))
    } else {
        Ok(id)
    }
}

/// Autoreleased `NSString`.
unsafe fn nsstring(s: &str) -> Result<Id, String> {
    let ns: Id = msg_send![class!(NSString), alloc];
    let ns: Id = msg_send![nonnull(ns, "NSString alloc")?, initWithBytes: s.as_ptr() as *const c_void length: s.len() encoding: 4usize];
    let ns: Id = msg_send![nonnull(ns, "NSString initWithBytes")?, autorelease];
    Ok(ns)
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

unsafe fn number(v: u64) -> Result<Id, String> {
    nonnull(
        msg_send![class!(NSNumber), numberWithUnsignedLongLong: v],
        "NSNumber",
    )
}

/// Autoreleased `NSData` copy of `bytes`.
unsafe fn data(bytes: &[u8]) -> Result<Id, String> {
    nonnull(
        msg_send![class!(NSData), dataWithBytes: bytes.as_ptr() as *const c_void length: bytes.len()],
        "NSData dataWithBytes",
    )
}

/// `NSMutableDictionary` of string keys; values must be non-nil (checked by their builders).
unsafe fn dict(pairs: &[(&str, Id)]) -> Result<Id, String> {
    let d = nonnull(
        msg_send![class!(NSMutableDictionary), dictionary],
        "NSMutableDictionary",
    )?;
    for &(k, v) in pairs {
        let (k, v) = (nsstring(k)?, nonnull(v, k)?);
        let () = msg_send![d, setObject: v forKey: k];
    }
    Ok(d)
}

unsafe fn array(items: &[Id]) -> Result<Id, String> {
    let a = nonnull(msg_send![class!(NSMutableArray), array], "NSMutableArray")?;
    for &v in items {
        let () = msg_send![a, addObject: nonnull(v, "array element")?];
    }
    Ok(a)
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
        if bytes == 0 {
            return Err("IOSurface of 0 bytes".into());
        }
        let raw = objc::rc::autoreleasepool(|| unsafe {
            let n = number(bytes as u64)?;
            let props = dict(&[
                ("IOSurfaceWidth", n),
                ("IOSurfaceHeight", number(1)?),
                ("IOSurfaceBytesPerElement", number(1)?),
                ("IOSurfaceBytesPerRow", n),
                ("IOSurfaceAllocSize", n),
                ("IOSurfacePixelFormat", number(0)?),
            ])?;
            Ok::<_, String>(IOSurfaceCreate(props))
        })?;
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

/// Programs of this process using each model directory (`$TMPDIR/<hexStringIdentifier>`).
/// Identical MIL and weights share one directory; the last program to drop removes the files
/// it wrote. Compiled models live in the framework's own cache, so later compiles of the same
/// program (in any process) still find them through `compiledModelExists`.
static MODEL_DIRS: LazyLock<Mutex<HashMap<PathBuf, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// One program's claim on its model directory.
struct ModelDir {
    dir: PathBuf,
}

impl ModelDir {
    /// Claims `$TMPDIR/<hex>` and writes `model.mil` and `weights/weight.bin` into it.
    fn claim(hex: &str, mil: &str, weights: &[u8]) -> Result<Self, String> {
        if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err(format!("ANE: unexpected model identifier {hex:?}"));
        }
        // SAFETY: NSTemporaryDirectory returns an autoreleased NSString (or nil).
        let tmp = unsafe { string(NSTemporaryDirectory()) };
        if tmp.is_empty() {
            return Err("ANE: NSTemporaryDirectory() is empty".into());
        }
        let dir = Path::new(&tmp).join(hex);
        *MODEL_DIRS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .entry(dir.clone())
            .or_insert(0) += 1;
        let claim = Self { dir };
        claim.write(mil, weights)?;
        Ok(claim)
    }

    /// (Re)writes the files the compiler reads, each through a rename so a concurrent reader of
    /// the same program never sees a partial file.
    fn write(&self, mil: &str, weights: &[u8]) -> Result<(), String> {
        let io = |e: std::io::Error| format!("ANE model dir {}: {e}", self.dir.display());
        std::fs::create_dir_all(self.dir.join("weights")).map_err(io)?;
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let put = |name: &str, bytes: &[u8]| {
            let seq = SEQ.fetch_add(1, Ordering::Relaxed);
            let tmp = self
                .dir
                .join(format!("{name}.{}-{seq}.tmp", std::process::id()));
            std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, self.dir.join(name)))
        };
        put("model.mil", mil.as_bytes()).map_err(io)?;
        if !weights.is_empty() {
            put("weights/weight.bin", weights).map_err(io)?;
        }
        Ok(())
    }
}

impl Drop for ModelDir {
    fn drop(&mut self) {
        let mut dirs = MODEL_DIRS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(n) = dirs.get_mut(&self.dir) else {
            return;
        };
        *n -= 1;
        if *n > 0 {
            return;
        }
        dirs.remove(&self.dir);
        // Only the files written here; directories go only if nothing else is left in them.
        let _ = std::fs::remove_file(self.dir.join("weights/weight.bin"));
        let _ = std::fs::remove_file(self.dir.join("model.mil"));
        let _ = std::fs::remove_dir(self.dir.join("weights"));
        let _ = std::fs::remove_dir(&self.dir);
    }
}

/// Reserves one compile from the per-process budget.
fn reserve_compile() -> Result<(), String> {
    COMPILED
        .try_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
            (n < ANE_MAX_PROGRAMS).then_some(n + 1)
        })
        .map(|_| ())
        .map_err(|_| format!("ANE program budget reached ({ANE_MAX_PROGRAMS} compiles)"))
}

/// A compiled and loaded ANE program with one reusable request over its surfaces.
pub struct AneProgram {
    model: Id,
    request: Id,
    inputs: Vec<Surface>,
    outputs: Vec<Surface>,
    _dir: ModelDir,
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
            let mil_data = data(mil.as_bytes())?;
            let wdict = if weights.is_empty() {
                dict(&[])?
            } else {
                let entry = dict(&[("offset", number(0)?), ("data", data(weights)?)])?;
                dict(&[("@model_path/weights/weight.bin", entry)])?
            };
            let desc = nonnull(
                msg_send![class("_ANEInMemoryModelDescriptor")?, modelWithMILText: mil_data weights: wdict optionsPlist: std::ptr::null_mut::<Object>()],
                "modelWithMILText:weights:optionsPlist:",
            )?;
            let model = nonnull(
                msg_send![class("_ANEInMemoryModel")?, inMemoryModelWithDescriptor: desc],
                "inMemoryModelWithDescriptor:",
            )?;
            let hex: Id = msg_send![model, hexStringIdentifier];
            let dir = ModelDir::claim(&string(hex), mil, weights)?;
            let opts = dict(&[])?;
            let mut err: Id = std::ptr::null_mut();
            let cached: BOOL = msg_send![model, compiledModelExists];
            if cached == NO {
                // Counted whether or not it succeeds: every attempt consumes compiler resources.
                reserve_compile()?;
                let ok: BOOL = msg_send![model, compileWithQoS: QOS options: opts error: &mut err];
                if ok == NO {
                    return Err(format!("ANE compile: {}", describe(err)));
                }
            }
            let mut ok: BOOL = msg_send![model, loadWithQoS: QOS options: opts error: &mut err];
            if ok == NO {
                // A just-unloaded program's slot can take a moment to free, and another process
                // finishing the same program may have removed the files in between.
                std::thread::sleep(std::time::Duration::from_millis(100));
                dir.write(mil, weights)?;
                ok = msg_send![model, loadWithQoS: QOS options: opts error: &mut err];
            }
            if ok == NO {
                return Err(format!("ANE load: {}", describe(err)));
            }
            let unload = |why: String| {
                let mut e: Id = std::ptr::null_mut();
                let _: BOOL = msg_send![model, unloadWithQoS: QOS error: &mut e];
                why
            };
            let io_class = class("_ANEIOSurfaceObject")?;
            let wrap = |s: &Surface| -> Result<Id, String> {
                nonnull(
                    msg_send![io_class, objectWithIOSurface: s.raw],
                    "objectWithIOSurface:",
                )
            };
            let request = (|| -> Result<Id, String> {
                let ins = inputs.iter().map(wrap).collect::<Result<Vec<_>, _>>()?;
                let outs = outputs.iter().map(wrap).collect::<Result<Vec<_>, _>>()?;
                let in_idx = (0..n_in)
                    .map(|i| number(i as u64))
                    .collect::<Result<Vec<_>, _>>()?;
                let out_idx = (0..n_out)
                    .map(|i| number(i as u64))
                    .collect::<Result<Vec<_>, _>>()?;
                let nil = std::ptr::null_mut::<Object>();
                nonnull(
                    msg_send![class("_ANERequest")?,
                        requestWithInputs: array(&ins)?
                        inputIndices: array(&in_idx)?
                        outputs: array(&outs)?
                        outputIndices: array(&out_idx)?
                        weightsBuffer: nil
                        perfStats: nil
                        procedureIndex: number(0)?],
                    "_ANERequest",
                )
            })()
            .map_err(unload)?;
            let _: Id = msg_send![model, retain];
            let _: Id = msg_send![request, retain];
            Ok(Self {
                model,
                request,
                inputs,
                outputs,
                _dir: dir,
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
            let opts = dict(&[])?;
            let mut err: Id = std::ptr::null_mut();
            let ok: BOOL = match entry {
                EvalEntry::InMemory => {
                    msg_send![self.model, evaluateWithQoS: QOS options: opts request: self.request error: &mut err]
                }
                EvalEntry::RealTime | EvalEntry::Direct => {
                    let sel = if entry == EvalEntry::RealTime {
                        "evaluateRealTimeWithModel:options:request:error:"
                    } else {
                        "doEvaluateDirectWithModel:options:request:qos:error:"
                    };
                    let c = class("_ANEClient")?;
                    if c.metaclass()
                        .instance_method(sel!(sharedConnection))
                        .is_none()
                        || !responds(c, sel)
                    {
                        return Err(format!("ANE: _ANEClient lacks {sel}"));
                    }
                    let client = nonnull(msg_send![c, sharedConnection], "sharedConnection")?;
                    let inner = nonnull(msg_send![self.model, model], "_ANEInMemoryModel model")?;
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
    }
}
