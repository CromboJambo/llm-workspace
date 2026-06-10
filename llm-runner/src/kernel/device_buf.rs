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

/// Host-backed buffer for data that lives on the CPU side.
#[derive(Clone)]
pub struct HostBuffer<T>(pub Vec<T>);

/// Device-backed buffer for data that lives on the GPU.
#[derive(Clone)]
pub enum DeviceBuffer<T> {
    /// Host data (kept for backward compatibility).
    Host(Vec<T>),
    /// Raw device pointer (no ownership — caller manages lifetime).
    Device(u64, usize),
    /// Raw device pointer with TMA descriptor attached.
    DeviceTma(u64, usize, TmaDescriptor),
    /// Owned cuda-oxide device buffer (Phase 2 wiring).
    Cuda(Arc<cuda_core::DeviceBuffer<T>>),
    /// CPU-backed "device" buffer for testing when CUDA is unavailable.
    CpuDevice(Vec<T>),
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        match self {
            Self::Cuda(buf) => {
                drop(buf);
            }
            _ => {}
        }
    }
}

unsafe impl<T: Send> Send for DeviceBuffer<T> {}
unsafe impl<T: Send> Sync for DeviceBuffer<T> {}

impl<T: std::fmt::Debug> std::fmt::Debug for DeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Host(v) => f.debug_tuple("Host").field(v).finish(),
            Self::Device(ptr, len) => f.debug_tuple("Device").field(ptr).field(len).finish(),
            Self::DeviceTma(ptr, len, _desc) => {
                f.debug_tuple("DeviceTma").field(ptr).field(len).finish()
            }
            Self::Cuda(_) => f.debug_tuple("Cuda").field(&"<device-buffer>").finish(),
            Self::CpuDevice(_) => f.debug_tuple("CpuDevice").field(&"<cpu-buffer>").finish(),
        }
    }
}

impl<T: Default + Clone> HostBuffer<T> {
    pub fn from_host(data: Vec<T>) -> Self {
        Self(data)
    }

    pub fn zeros(len: usize) -> Self
    where
        T: Default + Clone,
    {
        Self(vec![T::default(); len])
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    pub fn to_host(self) -> Vec<T> {
        self.0
    }

    pub fn into_inner(self) -> Vec<T> {
        self.0
    }
}

impl<T: Default + Clone> From<Vec<T>> for HostBuffer<T> {
    fn from(v: Vec<T>) -> Self {
        Self::from_host(v)
    }
}

impl<T: Default + Clone> From<HostBuffer<T>> for Vec<T> {
    fn from(buf: HostBuffer<T>) -> Self {
        buf.0
    }
}

impl<T> DeviceBuffer<T> {
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
            Self::CpuDevice(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_slice(&self) -> Option<&[T]> {
        match self {
            Self::Host(v) => Some(v.as_slice()),
            Self::CpuDevice(v) => Some(v.as_slice()),
            Self::Cuda(_) | Self::Device(..) | Self::DeviceTma(..) => None,
        }
    }

    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        match self {
            Self::Host(v) => Some(v.as_mut_slice()),
            Self::CpuDevice(v) => Some(v.as_mut_slice()),
            Self::Cuda(_) => None,
            Self::Device(..) | Self::DeviceTma(..) => None,
        }
    }

    pub fn to_host(&self) -> Vec<T>
    where
        T: Clone,
    {
        match self {
            Self::Host(v) => v.clone(),
            Self::CpuDevice(v) => v.clone(),
            Self::Cuda(_) | Self::Device(..) | Self::DeviceTma(..) => vec![],
        }
    }

    pub fn to_host_from_device(
        &self,
        stream: &cuda_core::CudaStream,
    ) -> Result<Vec<T>, DeviceBufferError>
    where
        T: Clone + cuda_core::DeviceCopy,
    {
        match self {
            Self::Cuda(buf) => buf
                .to_host_vec(stream)
                .map_err(|e| DeviceBufferError::Transfer(e.to_string())),
            Self::Host(v) => Ok(v.clone()),
            Self::CpuDevice(v) => Ok(v.clone()),
            Self::Device(..) | Self::DeviceTma(..) => Ok(vec![]),
        }
    }

    pub fn device_ptr(&self) -> Option<u64> {
        match self {
            Self::Device(ptr, _) | Self::DeviceTma(ptr, _, _) => Some(*ptr),
            Self::Cuda(buf) => Some(buf.cu_deviceptr()),
            Self::CpuDevice(_) => Some(0xDEAD), // Sentinel for CPU-fallback
            Self::Host(..) => None,
        }
    }

    pub fn tma_descriptor(&self) -> Option<TmaDescriptor> {
        match self {
            Self::DeviceTma(_, _, desc) => Some(*desc),
            _ => None,
        }
    }

    pub fn is_device(&self) -> bool {
        matches!(
            self,
            Self::Device(..) | Self::DeviceTma(..) | Self::Cuda(..) | Self::CpuDevice(..)
        )
    }

    pub fn as_cuda(&self) -> Option<&Arc<cuda_core::DeviceBuffer<T>>> {
        match self {
            Self::Cuda(buf) => Some(buf),
            _ => None,
        }
    }

    /// Create a device buffer from a raw pointer address and element count.
    pub unsafe fn from_device(ptr_addr: u64, len: usize) -> Self {
        Self::Device(ptr_addr, len)
    }

    /// Create a device buffer with an attached TMA descriptor.
    pub unsafe fn from_device_with_tma(ptr_addr: u64, len: usize, desc: TmaDescriptor) -> Self {
        Self::DeviceTma(ptr_addr, len, desc)
    }

    /// Attach a TMA descriptor to this device buffer.
    pub fn with_tma_descriptor(&self, desc: TmaDescriptor) -> Option<Self> {
        match self {
            Self::Device(ptr, len) => {
                Some(unsafe { Self::from_device_with_tma(*ptr, *len, desc) })
            }
            _ => None,
        }
    }

    /// Allocate device memory and copy data from host.
    pub fn from_host_device(
        stream: &cuda_core::CudaStream,
        data: &[T],
    ) -> Result<Self, DeviceBufferError>
    where
        T: Clone + Default + cuda_core::DeviceCopy,
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
        T: Clone + Default + cuda_core::DeviceCopy,
    {
        let buf = cuda_core::DeviceBuffer::zeroed(stream, len)
            .map_err(|e| DeviceBufferError::Allocation(e.to_string()))?;
        Ok(Self::Cuda(Arc::new(buf)))
    }

    /// Create a CPU-fallback device buffer from a host Vec.
    ///
    /// Used when CUDA is unavailable for testing. Data stays on the CPU
    /// but is treated as if it were on the device.
    pub fn from_cpu_device(data: Vec<T>) -> Self {
        Self::CpuDevice(data)
    }

    /// Create a zero-initialized CPU-fallback device buffer.
    pub fn zeros_cpu_device(len: usize) -> Self
    where
        T: Default + Clone,
    {
        Self::CpuDevice(vec![T::default(); len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_buffer_from_vec() {
        let buf = HostBuffer::from_host(vec![1, 2, 3]);
        assert_eq!(buf.len(), 3);
        assert!(!buf.is_empty());
        assert_eq!(buf.as_slice(), &[1, 2, 3]);
        assert_eq!(buf.to_host(), vec![1, 2, 3]);
    }

    #[test]
    fn host_buffer_zeros() {
        let buf: HostBuffer<i32> = HostBuffer::zeros(5);
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.to_host(), vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn host_buffer_empty() {
        let buf: HostBuffer<i32> = HostBuffer::from_host(vec![]);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn host_buffer_as_mut_slice() {
        let mut buf = HostBuffer::from_host(vec![1, 2, 3]);
        buf.as_mut_slice()[0] = 10;
        assert_eq!(buf.to_host(), vec![10, 2, 3]);
    }

    #[test]
    fn device_buffer_ptr() {
        let device_buf: DeviceBuffer<i32> = unsafe { DeviceBuffer::from_device(0x1000, 10) };
        assert_eq!(device_buf.device_ptr(), Some(0x1000));
        assert_eq!(device_buf.len(), 10);
        assert!(device_buf.is_device());
        assert!(device_buf.as_slice().is_none());
    }

    #[test]
    fn host_buffer_into_vec() {
        let buf = HostBuffer::from_host(vec![4, 5, 6]);
        let vec: Vec<i32> = buf.into_inner();
        assert_eq!(vec, vec![4, 5, 6]);
    }

    #[test]
    fn host_buffer_from_vec_trait() {
        let buf: HostBuffer<i32> = vec![7, 8, 9].into();
        assert_eq!(buf.to_host(), vec![7, 8, 9]);
    }

    #[test]
    fn cpu_device_buffer_from_vec() {
        let buf = DeviceBuffer::from_cpu_device(vec![1, 2, 3]);
        assert!(buf.is_device());
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.device_ptr(), Some(0xDEAD));
        let slice: &[i32] = buf.as_slice().unwrap();
        assert_eq!(slice, &[1, 2, 3]);
        assert_eq!(buf.to_host(), vec![1, 2, 3]);
    }

    #[test]
    fn cpu_device_buffer_zeros() {
        let buf: DeviceBuffer<i32> = DeviceBuffer::zeros_cpu_device(5);
        assert!(buf.is_device());
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.to_host(), vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn cpu_device_buffer_as_mut_slice() {
        let mut buf = DeviceBuffer::from_cpu_device(vec![1, 2, 3]);
        buf.as_mut_slice().unwrap()[0] = 10;
        assert_eq!(buf.to_host(), vec![10, 2, 3]);
    }

    #[test]
    fn cpu_device_buffer_is_not_cuda() {
        let buf = DeviceBuffer::from_cpu_device(vec![1, 2, 3]);
        assert!(buf.as_cuda().is_none());
    }
}
