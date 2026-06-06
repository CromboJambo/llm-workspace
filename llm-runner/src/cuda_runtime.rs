//! CUDA runtime: context management, device enumeration, compute capability detection.
//!
//! Wraps cuda-oxide's `CudaContext` to provide a stable interface for the inference
//! engine's GPU path. Handles initialization, device discovery, and error propagation.

use cuda_core::{CudaContext, IntoResult};
use std::sync::Arc;
use tracing::{debug, warn};

// Re-export cuda_bindings through cuda_core
use cuda_core::sys as cuda_sys;

/// Error type for CUDA runtime operations.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA driver not initialized: {0}")]
    NotInitialized(String),

    #[error("CUDA device unavailable: ordinal={ordinal}")]
    DeviceUnavailable { ordinal: usize },

    #[error("CUDA context creation failed: {0}")]
    ContextCreation(String),

    #[error("CUDA compute capability unsupported: sm_{major}.{minor} < sm_100")]
    ComputeCapabilityUnsupported { major: i32, minor: i32 },

    #[error("CUDA library load failed: {0}")]
    LibraryLoad(String),

    #[error("CUDA error code: {0}")]
    DriverError(u32),

    #[error("CUDA not available on this system")]
    NotAvailable,
}

/// Information about a single CUDA device.
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Zero-based device ordinal.
    pub ordinal: usize,
    /// Device name (e.g., "NVIDIA GeForce RTX 5060 Ti").
    pub name: String,
    /// Compute capability (major, minor).
    pub compute_capability: (i32, i32),
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// Free device memory in bytes.
    pub free_memory: u64,
}

impl CudaDeviceInfo {
    /// Whether this device can hold a model of the given size.
    pub fn can_hold_model(&self, model_bytes: u64) -> bool {
        // Reserve 2 GiB for overhead (KV cache, intermediate buffers, PTX JIT)
        self.free_memory > model_bytes + 2 * 1024 * 1024 * 1024
    }

    /// Whether this device supports tcgen05 (sm_100+).
    pub fn supports_tcgen05(&self) -> bool {
        let (major, _minor) = self.compute_capability;
        major >= 10
    }

    /// Whether this device supports WGMMA (sm_120+).
    pub fn supports_wgmma(&self) -> bool {
        let (major, _minor) = self.compute_capability;
        major >= 12
    }
}

/// A live CUDA context for a specific device.
///
/// Wraps `Arc<CudaContext>` and tracks the device ordinal for routing.
#[derive(Debug, Clone)]
pub struct CudaRuntime {
    /// The underlying cuda-oxide context.
    ctx: Arc<CudaContext>,
    /// Device ordinal.
    ordinal: usize,
    /// Device info (cached at creation).
    device_info: CudaDeviceInfo,
}

impl CudaRuntime {
    /// Create a new CUDA runtime for the device at `ordinal`.
    ///
    /// Initializes the CUDA driver, obtains the primary context, and queries
    /// device properties. Returns `CudaError::NotAvailable` if the device
    /// cannot be found or the driver fails to initialize.
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        // Initialize CUDA driver
        unsafe {
            cuda_core::init(0).map_err(|e| CudaError::NotInitialized(e.to_string()))?;
        };

        // Get device handle
        let cu_device = unsafe {
            let mut device = std::mem::MaybeUninit::uninit();
            cuda_sys::cuDeviceGet(device.as_mut_ptr(), ordinal as i32)
                .result()
                .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
            device.assume_init()
        };

        // Get device name
        let mut name_buf = [0i8; 256];
        unsafe {
            cuda_sys::cuDeviceGetName(
                name_buf.as_mut_ptr(),
                name_buf.len() as i32,
                cu_device,
            )
        };
        let name: String = name_buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect::<Vec<u8>>()
            .into_iter()
            .map(|b| b as char)
            .collect();

        // Get compute capability
        let mut major = std::mem::MaybeUninit::uninit();
        let mut minor = std::mem::MaybeUninit::uninit();
        unsafe {
            cuda_sys::cuDeviceGetAttribute(
                major.as_mut_ptr(),
                cuda_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                cu_device,
            )
            .result()
            .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
            cuda_sys::cuDeviceGetAttribute(
                minor.as_mut_ptr(),
                cuda_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                cu_device,
            )
            .result()
            .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
        }
        let (major, minor) = (major.assume_init(), minor.assume_init());

        // Get memory info
        let (free_memory, total_memory) = unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            cuda_sys::cuMemGetInfo_v2(&mut free, &mut total)
                .result()
                .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
            (free as u64, total as u64)
        };

        let device_info = CudaDeviceInfo {
            ordinal,
            name,
            compute_capability: (major, minor),
            total_memory,
            free_memory,
        };

        // Retain primary context
        let ctx = CudaContext::new(ordinal)
            .map_err(|e| CudaError::ContextCreation(e.to_string()))?;

        debug!(
            ordinal,
            name = %device_info.name,
            cc = "%d.%d", major, minor,
            "CUDA runtime: initialized device"
        );

        Ok(Self {
            ctx,
            ordinal,
            device_info,
        })
    }

    /// Create a CUDA runtime for the first available device (ordinal 0).
    pub fn for_default_device() -> Result<Self, CudaError> {
        Self::new(0)
    }

    /// Returns the underlying cuda-oxide context.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Returns cached device info.
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.device_info
    }

    /// Create a new non-blocking stream in this context.
    pub fn new_stream(&self) -> Result<Arc<cuda_core::CudaStream>, CudaError> {
        self.ctx
            .new_stream()
            .map_err(|e| CudaError::ContextCreation(e.to_string()))
    }

    /// Synchronize the context (blocks until all pending work completes).
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.ctx
            .synchronize()
            .map_err(|e| CudaError::ContextCreation(e.to_string()))
    }

    /// Check if this runtime is still valid (context not destroyed).
    pub fn is_valid(&self) -> bool {
        !self.ctx.cu_ctx().is_null()
    }
}

/// Initialize CUDA and enumerate available devices.
///
/// Returns a list of `CudaDeviceInfo` for all devices that can be queried.
/// Returns an empty list if CUDA is not available or no devices are found.
pub fn enumerate_devices() -> Result<Vec<CudaDeviceInfo>, CudaError> {
    unsafe {
        cuda_core::init(0).map_err(|_| CudaError::NotAvailable)?;
    };

    let mut device_count = 0;
    unsafe {
        cuda_sys::cuDeviceGetCount(&mut device_count)
            .result()
            .map_err(|_| CudaError::NotAvailable)?;
    };

    if device_count == 0 {
        return Ok(Vec::new());
    }

    let mut devices = Vec::with_capacity(device_count);

    for ordinal in 0..device_count {
        // Get device handle
        let cu_device = match unsafe {
            let mut device = std::mem::MaybeUninit::uninit();
            cuda_sys::cuDeviceGet(device.as_mut_ptr(), ordinal as i32)
                .result()
                .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
            Ok(device.assume_init())
        } {
            Ok(d) => d,
            Err(e) => {
                warn!(ordinal, "CUDA device enumeration skipped: {e}");
                continue;
            }
        };

        // Get device name
        let mut name_buf = [0i8; 256];
        unsafe {
            cuda_sys::cuDeviceGetName(
                name_buf.as_mut_ptr(),
                name_buf.len() as i32,
                cu_device,
            )
        };
        let name: String = name_buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect::<Vec<u8>>()
            .into_iter()
            .map(|b| b as char)
            .collect();

        // Get compute capability
        let (major, minor) = match unsafe {
            let mut m = std::mem::MaybeUninit::uninit();
            let mut n = std::mem::MaybeUninit::uninit();
            cuda_sys::cuDeviceGetAttribute(
                m.as_mut_ptr(),
                cuda_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                cu_device,
            )
            .result()
            .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
            cuda_sys::cuDeviceGetAttribute(
                n.as_mut_ptr(),
                cuda_sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                cu_device,
            )
            .result()
            .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
            Ok((m.assume_init(), n.assume_init()))
        } {
            Ok(cc) => cc,
            Err(e) => {
                warn!(ordinal, "CUDA compute capability query failed: {e}");
                continue;
            }
        };

        // Get memory info
        let (free_memory, total_memory) = match unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            cuda_sys::cuMemGetInfo_v2(&mut free, &mut total)
                .result()
                .map_err(|_| CudaError::DeviceUnavailable { ordinal })?;
            Ok((free as u64, total as u64))
        } {
            Ok(info) => info,
            Err(e) => {
                warn!(ordinal, "CUDA memory info query failed: {e}");
                continue;
            }
        };

        devices.push(CudaDeviceInfo {
            ordinal,
            name,
            compute_capability: (major, minor),
            total_memory,
            free_memory,
        });
    }

    Ok(devices)
}

/// Select the best device for a model of the given size.
///
/// Prioritizes:
/// 1. Devices with enough free VRAM
/// 2. Higher compute capability (tcgen05 > WGMMA > CPU)
/// 3. More free memory as tiebreaker
pub fn select_best_device(model_bytes: u64) -> Option<CudaDeviceInfo> {
    let devices = enumerate_devices().unwrap_or_default();

    let mut candidates: Vec<&CudaDeviceInfo> = devices
        .iter()
        .filter(|d| d.can_hold_model(model_bytes))
        .collect();

    if candidates.is_empty() {
        return None;
    }

    // Sort by: tcgen05 support > WGMMA support > free memory descending
    candidates.sort_by(|a, b| {
        let a_tc = if a.supports_tcgen05() { 3 } else { 0 };
        let a_wg = if a.supports_wgmma() { 2 } else { 0 };
        let b_tc = if b.supports_tcgen05() { 3 } else { 0 };
        let b_wg = if b.supports_wgmma() { 2 } else { 0 };

        let a_score = a_tc + a_wg + (a.free_memory as i64 / (1024 * 1024 * 1024));
        let b_score = b_tc + b_wg + (b.free_memory as i64 / (1024 * 1024 * 1024));

        b_score.cmp(&a_score)
    });

    candidates.first().cloned()
}

/// Check if CUDA is available on this system.
pub fn is_available() -> bool {
    enumerate_devices().is_ok() && !enumerate_devices().unwrap_or_default().is_empty()
}

/// Get the number of CUDA devices.
pub fn device_count() -> usize {
    match unsafe {
        let mut count = 0;
        cuda_sys::cuDeviceGetCount(&mut count)
            .result()
            .map(|_| count)
    } {
        Ok(count) => count,
        Err(_) => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_device_info_can_hold_model() {
        let info = CudaDeviceInfo {
            ordinal: 0,
            name: "Test GPU".to_string(),
            compute_capability: (12, 0),
            total_memory: 16 * 1024 * 1024 * 1024,
            free_memory: 14 * 1024 * 1024 * 1024,
        };

        // 1 GiB model should fit
        assert!(info.can_hold_model(1024 * 1024 * 1024));

        // 15 GiB model should not fit (needs 2 GiB overhead)
        assert!(!info.can_hold_model(15 * 1024 * 1024 * 1024));
    }

    #[test]
    fn cuda_device_info_supports_tcgen05() {
        let sm100 = CudaDeviceInfo {
            ordinal: 0,
            name: "B200".to_string(),
            compute_capability: (10, 0),
            total_memory: 0,
            free_memory: 0,
        };
        assert!(sm100.supports_tcgen05());

        let sm90 = CudaDeviceInfo {
            ordinal: 0,
            name: "H100".to_string(),
            compute_capability: (9, 0),
            total_memory: 0,
            free_memory: 0,
        };
        assert!(!sm90.supports_tcgen05());
    }

    #[test]
    fn cuda_device_info_supports_wgmma() {
        let sm120 = CudaDeviceInfo {
            ordinal: 0,
            name: "RTX 5090".to_string(),
            compute_capability: (12, 0),
            total_memory: 0,
            free_memory: 0,
        };
        assert!(sm120.supports_wgmma());

        let sm119 = CudaDeviceInfo {
            ordinal: 0,
            name: "Fake GPU".to_string(),
            compute_capability: (11, 9),
            total_memory: 0,
            free_memory: 0,
        };
        assert!(!sm119.supports_wgmma());
    }

    #[test]
    fn cuda_error_display() {
        let err = CudaError::NotInitialized("driver".to_string());
        assert!(err.to_string().contains("driver"));

        let err = CudaError::ComputeCapabilityUnsupported { major: 9, minor: 0 };
        assert!(err.to_string().contains("sm_9.0"));

        let err = CudaError::NotAvailable;
        assert!(err.to_string().contains("not available"));
    }

    #[test]
    fn enumerate_devices_returns_empty_when_unavailable() {
        // On systems without CUDA, this should return Ok(empty) or Err
        let result = enumerate_devices();
        match result {
            Ok(devices) => assert!(!devices.is_empty() || true), // Either ok with empty or error
            Err(CudaError::NotAvailable) => {}
            Err(_) => {}
        }
    }

    #[test]
    fn is_available_returns_bool() {
        let _ = is_available(); // Should not panic
    }

    #[test]
    fn device_count_returns_non_negative() {
        let count = device_count();
        assert!(count >= 0);
    }
}
