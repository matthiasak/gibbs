#[cfg(target_os = "macos")]
pub mod metal {
    pub use crate::graph::metal::{
        commit_and_wait, new_shared_buffer, system_device, system_device_name,
        system_has_unified_memory,
    };
}
