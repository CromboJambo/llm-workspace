//! Device buffer abstraction for host and device memory.
//!
//! Unified interface over Vec<T> (host) and cuda-oxide DeviceBuffer<T> (device).
//! Used by GEMM kernels for A, B, C matrices and KV cache.
//!
//! Device buffers support TMA descriptor attachment for Blackwell async GMEM→SMEM copies.

use crate::kernel::tma_descriptor::TmaDescriptor;
use std::sync::Arc;

/// Error type for device buffer operations.
#[derive(Debug, thiserror::Error)]
pub enum DeviceBufferError {
    #[error("CUDA context not available")]
    NoContext,

    #[error("CUDA allocation failed: {0}")]
    Allocation(String),

    #[error("CUDA transfer failed: {0}")]
    Transfer(String),

    #[error("buffer size mismatch: expected {expected}, got {got}")]
    SizeMismatch { expected: usize, got: usize },
}

#[derive(Clone)]
pub enum DeviceBuffer<T> {
    Host(Vec<T>),
    Device(u64, usize), // ptr_addr, len_elements
    DeviceTma(u64, usize, TmaDescriptor), // ptr_addr, len_elements, tma_descriptor
    /// Real cuda-oxide backed device buffer (Phase 2 wiring).
    Cuda(Arc<cuda_core::DeviceBuffer<T>>),
}

impl<T: std::fmt::Debug> std::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host(v) => f.debug_tuple("Host").field(v).finish(),
            Self::Device(ptr, len) => f.debug_tuple("Device").field(ptr).field(len).finish(),
            Self::DeviceTma(ptr, len, _desc) => f.debug_tuple("DeviceTma").field(ptr).field(len).finish(),
            Self::Cuda(_) => f.debug_tuple("Cuda").field(&"<device-buffer>").finish(),
        }
    }
}

impl<T: Default + Clone> DeviceBuffer<T> {
    pub fn from_host(data: Vec<T>) -> Self {
        Self::Host(data)
    }

    pub fn zeros(len: usize) -> Self
    where
        T: Default + Clone,
    {
        Self::Host(vec![T::default(); len])
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Host(v) => v.len(),
            Self::Device(_, len) | Self::DeviceTma(_, len, _) => *len,
            Self::Cuda(buf) => buf.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> Option<&[T]> {
        match self {
            Self::Host(v) => Some(v.as_slice()),
            Self::Cuda(_) | Self::Device(..) | Self::DeviceTma(..) => None,
        }
    }

    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        match self {
            Self::Host(v) => Some(v.as_mut_slice()),
            Self::Cuda(_) => None, // Can't get mutable ref to device buffer
            Self::Device(..) | Self::DeviceTma(..) => None,
        }
    }

    pub fn to_host(&self) -> Vec<T>
    where
        T: Clone,
    {
        match self {
            Self::Host(v) => v.clone(),
            Self::Cuda(_) | Self::Device(..) | Self::DeviceTma(..) => vec![],
        }
    }
}

impl<T> DeviceBuffer<T> {
    /// Create a device buffer from a raw pointer address and element count.
    ///
    /// # Safety
    ///
    /// Caller must ensure `ptr_addr` is valid and points to at least `len * size_of::<T>()` bytes.
    pub unsafe fn from_device(ptr_addr: u64, len: usize) -> Self {
        Self::Device(ptr_addr, len)
    }

    /// Create a device buffer with an attached TMA descriptor.
    ///
    /// # Safety
    ///
    /// Caller must ensure `ptr_addr` is valid and points to at least `len * size_of::<T>()` bytes.
    pub unsafe fn from_device_with_tma(ptr_addr: u64, len: usize, desc: TmaDescriptor) -> Self {
        Self::DeviceTma(ptr_addr, len, desc)
    }

    /// Attach a TMA descriptor to this device buffer.
    ///
    /// Returns a new `DeviceBuffer` with the descriptor attached.
    /// If this buffer is not on device, returns `None`.
    pub fn with_tma_descriptor(&self, desc: TmaDescriptor) -> Option<Self> {
        match self {
            Self::Device(ptr, len) => {
                // Safety: we're just attaching metadata, not dereferencing the pointer
                Some(unsafe { Self::from_device_with_tma(*ptr, *len, desc) })
            }
            _ => None,
        }
    }

    pub fn device_ptr(&self) -> Option<u64> {
        match self {
            Self::Device(ptr, _) | Self::DeviceTma(ptr, _, _) => Some(*ptr),
            Self::Cuda(buf) => Some(buf.cu_deviceptr()),
            Self::Host(..) => None,
        }
    }

    /// Get the TMA descriptor if this buffer has one attached.
    pub fn tma_descriptor(&self) -> Option<TmaDescriptor> {
        match self {
            Self::DeviceTma(_, _, desc) => Some(*desc),
            _ => None,
        }
    }

    /// Check if this buffer is on the device (not host).
    pub fn is_device(&self) -> bool {
        matches!(self, Self::Device(..) | Self::DeviceTma(..) | Self::Cuda(..))
    }

    /// Allocate device memory and copy data from host.
    ///
    /// Requires a live CUDA context. Returns `DeviceBufferError::NoContext` if CUDA is unavailable.
    pub fn from_host_device(
        stream: &cuda_core::CudaStream,
        data: &[T],
    ) -> Result<Self, DeviceBufferError>
    where
        T: Clone + Default,
    {
        let buf = cuda_core::DeviceBuffer::from_host(stream, data)
            .map_err(|e| DeviceBufferError::Allocation(e.to_string()))?;
        Ok(Self::Cuda(Arc::new(buf)))
    }

    /// Allocate zero-initialized device memory.
    pub fn zeros_device(
        stream: &cuda_core::CudaStream,
        len: usize,
    ) -> Result<Self, DeviceBufferError>
    where
        T: Clone + Default,
    {
        let buf = cuda_core::DeviceBuffer::zeroed(stream, len)
            .map_err(|e| DeviceBufferError::Allocation(e.to_string()))?;
        Ok(Self::Cuda(Arc::new(buf)))
    }

    /// Copy device buffer contents back to host.
    pub fn to_host_from_device(&self, stream: &cuda_core::CudaStream) -> Result<Vec<T>, DeviceBufferError>
    where
        T: Clone,
    {
        match self {
            Self::Cuda(buf) => buf
                .to_host_vec(stream)
                .map_err(|e| DeviceBufferError::Transfer(e.to_string())),
            Self::Host(v) => Ok(v.clone()),
            Self::Device(..) | Self::DeviceTma(..) => Ok(vec![]), // Can't read from raw ptr
        }
    }

    /// Get the underlying cuda-oxide buffer if this is a Cuda variant.
    pub fn as_cuda(&self) -> Option<&Arc<cuda_core::DeviceBuffer<T>>> {
        match self {
            Self::Cuda(buf) => Some(buf),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_host() {
        let buf = DeviceBuffer::from_host(vec![1, 2, 3]);
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_empty());
        assert_eq!(buf.as_slice(), Some(&[1, 2, 3][..]));
        assert_eq!(buf.to_host(), vec![1, 2, 3]);
    }

    #[test]
    fn zeros() {
        let buf: DeviceBuffer<i32> = DeviceBuffer::zeros(5);
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.to_host(), vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn empty() {
        let buf: DeviceBuffer<i32> = DeviceBuffer::from_host(vec![]);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn as_mut_slice() {
        let mut buf = DeviceBuffer::from_host(vec![1, 2, 3]);
        if let Some(slice) = buf.as_mut_slice() {
            slice[0] = 10;
        }
        assert_eq!(buf.to_host(), vec![10, 2, 3]);
    }

    #[test]
    fn device_ptr() {
        let host_buf = DeviceBuffer::from_host(vec![1]);
        assert!(host_buf.device_ptr().is_none());

        let device_buf: DeviceBuffer<i32> = unsafe { DeviceBuffer::from_device(0x1000, 10) };
        assert_eq!(device_buf.device_ptr(), Some(0x1000));
        assert_eq!(device_buf.len(), 10);
        assert!(device_buf.as_slice().is_none());
    }
}
