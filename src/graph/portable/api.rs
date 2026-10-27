//! The device API a portable backend implements: memory, runtime compilation and kernel launch.
//! Handles are opaque `u64`s (a `CUdeviceptr`, a `hipDeviceptr_t`, a `cl_mem`, ...).

use super::dialect::Dialect;

/// A device allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Buf(pub u64);

/// A compiled program (`CUmodule`, `hipModule_t`, `cl_program`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Module(pub u64);

/// One entry point of a [`Module`] (`CUfunction`, `hipFunction_t`, `cl_kernel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Kernel(pub u64);

/// A captured launch sequence (`CUgraphExec`, `hipGraphExec_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Replay(pub u64);

/// One kernel argument: a device buffer or a 4-/8-byte scalar, in kernel parameter order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Arg {
    Buf(Buf),
    U32(u32),
    F32(f32),
    U64(u64),
}

/// Device memory, compilation and launch. Every call is checked; errors name the call and code.
/// Launches and copies are ordered (one in-order queue / stream per API object); `download`
/// returns after the data is on the host.
pub trait GpuApi: Send + Sync {
    fn device_name(&self) -> &str;
    fn dialect(&self) -> Dialect;
    /// `bytes` (> 0) of zero-filled device memory.
    fn alloc(&self, bytes: usize) -> Result<Buf, String>;
    fn free(&self, buf: Buf);
    fn upload(&self, buf: Buf, offset: usize, data: &[u8]) -> Result<(), String>;
    fn download(&self, buf: Buf, offset: usize, out: &mut [u8]) -> Result<(), String>;
    /// Compiles one complete translation unit (the dialect prelude included).
    fn compile(&self, src: &str) -> Result<Module, String>;
    fn kernel(&self, module: &Module, name: &str) -> Result<Kernel, String>;
    /// Launches `grid` workgroups of `block` threads each.
    fn launch(
        &self,
        k: Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[Arg],
    ) -> Result<(), String>;
    /// Waits until every launch and copy issued so far has finished.
    fn sync(&self) -> Result<(), String>;
    /// Starts recording launches instead of running them; `false` when the API cannot replay.
    fn capture_begin(&self) -> Result<bool, String> {
        Ok(false)
    }
    /// Ends a [`Self::capture_begin`] recording and instantiates it.
    fn capture_end(&self) -> Result<Replay, String> {
        Err("capture is not supported".into())
    }
    /// Runs a recorded launch sequence.
    fn replay(&self, _r: Replay) -> Result<(), String> {
        Err("replay is not supported".into())
    }
    fn replay_free(&self, _r: Replay) {}
}
