//! Local GPU device discovery.
//!
//! Enumerates available CUDA GPUs with VRAM, compute capability, and health status.
//! Uses cudarc driver API for device enumeration.

use serde::{Deserialize, Serialize};
use tracing::debug;

/// A discovered local GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalDevice {
    /// CUDA device ordinal.
    pub ordinal: u32,
    /// GPU name (e.g., "NVIDIA GeForce RTX 4070").
    pub name: String,
    /// Total VRAM in bytes.
    pub total_vram: u64,
    /// Free VRAM in bytes (approximate, may change).
    pub free_vram: u64,
    /// Compute capability (major.minor).
    pub compute_capability: String,
    /// Whether the device is currently available.
    pub available: bool,
    /// Used VRAM in bytes (if detectable).
    pub used_vram: u64,
}

impl LocalDevice {
    /// Free VRAM as a percentage of total.
    pub fn free_vram_pct(&self) -> f32 {
        if self.total_vram == 0 {
            return 0.0;
        }
        (self.free_vram as f32 / self.total_vram as f32) * 100.0
    }

    /// Model size in bytes that can fit on this device.
    ///
    /// Conservative: reserves 2GiB for CUDA context and kernels.
    pub fn max_model_bytes(&self) -> u64 {
        // Reserve 2GiB for CUDA context, kernels, KV cache overhead
        let reserve = 2 * 1024 * 1024 * 1024u64;
        if self.free_vram > reserve {
            self.free_vram - reserve
        } else {
            0
        }
    }

    /// Whether this device can hold a model of the given size.
    pub fn can_hold_model(&self, model_bytes: u64) -> bool {
        self.max_model_bytes() >= model_bytes
    }
}

/// Discover all local CUDA devices.
///
/// Returns an empty vec if no CUDA-capable GPU is available.
pub fn discover_local_devices() -> Vec<LocalDevice> {
    let mut devices = Vec::new();

    // Try to enumerate CUDA devices via cudarc
    match initialize_cuda() {
        Ok(_) => {
            if let Ok(count) = get_device_count() {
                for ordinal in 0..count {
                    if let Some(device) = get_device_info(ordinal) {
                        devices.push(device);
                        debug!(ordinal, name = %device.name, "Discovered local CUDA device");
                    }
                }
            }
        }
        Err(e) => {
            debug!(error = %e, "CUDA initialization failed, no local GPUs discovered");
        }
    }

    // Always include CPU as fallback
    devices.push(LocalDevice {
        ordinal: u32::MAX,
        name: "CPU (fallback)".to_string(),
        total_vram: 0,
        free_vram: 0,
        compute_capability: "N/A".to_string(),
        available: true,
        used_vram: 0,
    });

    debug!(count = devices.len(), "Local device discovery complete");
    devices
}

/// Initialize CUDA driver API.
fn initialize_cuda() -> Result<(), String> {
    // cudarc driver API init
    // We use the raw driver API through cudarc
    unsafe {
        cudarc::driver::result::init()
            .map_err(|e| format!("CUDA init failed: {e}"))
    }
}

/// Get the number of CUDA devices.
fn get_device_count() -> Result<u32, String> {
    unsafe {
        cudarc::driver::result::device::get_count()
            .map(|c| c as u32)
            .map_err(|e| format!("Device count failed: {e}"))
    }
}

/// Get info for a specific CUDA device.
fn get_device_info(ordinal: u32) -> Option<LocalDevice> {
    use cudarc::driver::result::device::GetInfo;

    unsafe {
        // Get device properties
        let name = match cudarc::driver::result::device::get_name(ordinal) {
            Ok(n) => n,
            Err(_) => return None,
        };

        // Get compute capability
        let major = match cudarc::driver::result::device::get_compute_major(ordinal) {
            Ok(m) => m,
            Err(_) => return None,
        };
        let minor = match cudarc::driver::result::device::get_compute_minor(ordinal) {
            Ok(m) => m,
            Err(_) => return None,
        };

        // Get memory info
        let total = match cudarc::driver::result::memory::get_allocations(ordinal) {
            Ok((total, free)) => {
                let free_vram = free;
                Some((total, free_vram))
            }
            Err(_) => None,
        }?;

        let (total_vram, free_vram) = total;
        let used_vram = total_vram.saturating_sub(free_vram);

        // Check if device is accessible
        let available = cudarc::driver::result::device::get_persistence_mode(ordinal)
            .map(|_| true)
            .unwrap_or(false);

        Some(LocalDevice {
            ordinal,
            name,
            total_vram,
            free_vram,
            compute_capability: format!("{major}.{minor}"),
            available,
            used_vram,
        })
    }
}

/// Get the best available local GPU for inference.
///
/// Returns the device with the most free VRAM that can hold the model,
/// or None if no GPU can fit the model.
pub fn select_best_gpu(model_bytes: u64) -> Option<LocalDevice> {
    let devices = discover_local_devices();
    let gpus: Vec<_> = devices
        .into_iter()
        .filter(|d| d.ordinal != u32::MAX)
        .filter(|d| d.can_hold_model(model_bytes))
        .collect();

    // Return GPU with most free VRAM
    gpus.into_iter()
        .max_by_key(|d| d.free_vram)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_device_free_vram_pct() {
        let device = LocalDevice {
            ordinal: 0,
            name: "Test GPU".to_string(),
            total_vram: 16_000_000_000,
            free_vram: 8_000_000_000,
            compute_capability: "8.0".to_string(),
            available: true,
            used_vram: 8_000_000_000,
        };
        assert!((device.free_vram_pct() - 50.0).abs() < 0.1);
    }

    #[test]
    fn local_device_max_model_bytes() {
        let device = LocalDevice {
            ordinal: 0,
            name: "Test GPU".to_string(),
            total_vram: 16_000_000_000,
            free_vram: 10_000_000_000,
            compute_capability: "8.0".to_string(),
            available: true,
            used_vram: 6_000_000_000,
        };
        // Reserves 2GiB
        let expected = 10_000_000_000 - (2 * 1024 * 1024 * 1024);
        assert_eq!(device.max_model_bytes(), expected);
    }

    #[test]
    fn local_device_max_model_bytes_zero_reserve() {
        let device = LocalDevice {
            ordinal: 0,
            name: "Test GPU".to_string(),
            total_vram: 4_000_000_000,
            free_vram: 1_000_000_000,
            compute_capability: "8.0".to_string(),
            available: true,
            used_vram: 3_000_000_000,
        };
        // Free VRAM < reserve, returns 0
        assert_eq!(device.max_model_bytes(), 0);
    }

    #[test]
    fn local_device_can_hold_model() {
        let device = LocalDevice {
            ordinal: 0,
            name: "Test GPU".to_string(),
            total_vram: 16_000_000_000,
            free_vram: 10_000_000_000,
            compute_capability: "8.0".to_string(),
            available: true,
            used_vram: 6_000_000_000,
        };
        let small_model = 5_000_000_000u64;
        let large_model = 20_000_000_000u64;
        assert!(device.can_hold_model(small_model));
        assert!(!device.can_hold_model(large_model));
    }

    #[test]
    fn discover_local_devices_includes_cpu_fallback() {
        let devices = discover_local_devices();
        // CPU fallback is always included
        assert!(devices.iter().any(|d| d.ordinal == u32::MAX));
    }
}
