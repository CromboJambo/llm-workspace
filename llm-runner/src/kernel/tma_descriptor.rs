//! Blackwell TMA global cache read descriptor.
//!
//! The TMA global cache read descriptor is a 128-bit hardware structure
//! that encodes the source of an asynchronous GMEM-to-SMEM copy.
//!
//! The descriptor holds:
//! - GMEM address offset (word 0)
//! - Box dimensions (X and Y)
//! - GMEM and SMEM strides
//! - Element info, descriptor type, SMEM config, cache hint
//!
//! For the KV cache use case, the address offset is a byte offset from
//! the buffer base (passed separately to the kernel). The box defines
//! the region to copy, and strides define how to stride through GMEM/SMEM.

#[derive(Debug, Clone, Copy)]
pub struct TmaDescriptor(pub [u32; 4]);

impl TmaDescriptor {
    /// Create a new zeroed descriptor.
    pub const fn new() -> Self {
        Self([0u32; 4])
    }

    /// Set the GMEM address offset (within the buffer base).
    ///
    /// The offset must be 256-byte aligned for TMA.
    /// Stored in word 0 [31:0].
    pub const fn with_gmem_addr(mut self, addr: u64) -> Self {
        self.0[0] = (addr & 0xFFFFFFFF) as u32;
        self
    }

    /// Get the GMEM address offset from the descriptor.
    pub fn gmem_addr(&self) -> u64 {
        self.0[0] as u64
    }

    /// Set the box (region) dimensions and strides.
    ///
    /// `box_x` — number of elements along X axis (16-bit).
    /// `gmem_x_stride` — GMEM stride in elements between consecutive rows (8-bit, max 255).
    /// `smem_x_stride` — SMEM stride in elements between consecutive rows (8-bit, max 255).
    /// `box_y` — number of elements along Y axis (16-bit).
    /// `gmem_y_stride` — GMEM stride in elements between consecutive columns (16-bit).
    /// `smem_y_stride` — SMEM stride in elements between consecutive columns (16-bit).
    ///   Stored in word 3 [15:0] to avoid word boundary issues.
    #[allow(clippy::too_many_arguments)]
    pub fn with_box(
        mut self,
        box_x: u16,
        gmem_x_stride: u16,
        smem_x_stride: u16,
        box_y: u16,
        gmem_y_stride: u16,
        smem_y_stride: u16,
    ) -> Self {
        // Word 1: box X [15:0], gmem_x_stride [23:16], smem_x_stride [31:24]
        self.0[1] |= box_x as u32;
        self.0[1] |= ((gmem_x_stride.min(255)) as u32) << 16;
        self.0[1] |= ((smem_x_stride.min(255)) as u32) << 24;

        // Word 2: box Y [15:0], gmem_y_stride [31:16]
        self.0[2] |= box_y as u32;
        self.0[2] |= (gmem_y_stride as u32) << 16;

        // Word 3: smem_y_stride [15:0]
        self.0[3] |= (smem_y_stride as u32) << 16;

        self
    }

    /// Set the element info field.
    ///
    /// element_size: 0=1B, 1=2B, 2=4B, 3=8B. f16 = 2B → value 1.
    pub fn with_element_info(mut self, element_size: u8) -> Self {
        // Stored in word 1 bits [31:28]
        self.0[1] |= ((element_size as u32) & 0xF) << 28;
        self
    }

    /// Set the descriptor type.
    ///
    /// 0x01 = Global Cache Read (TMA load).
    pub fn with_descriptor_type(mut self, dtype: u8) -> Self {
        self.0[3] |= (dtype as u32) << 24;
        self
    }

    /// Set the SMEM config field.
    pub fn with_smem_config(mut self, config: u8) -> Self {
        self.0[3] |= (config as u32) << 28;
        self
    }

    /// Set the cache hint.
    ///
    /// 0 = default (read-through to L2).
    /// 1 = don't allocate in L2.
    pub fn with_cache_hint(mut self, hint: u8) -> Self {
        self.0[3] |= (hint as u32) << 30;
        self
    }

    /// Pack the descriptor into bytes for passing to `tmca_gcr_read`.
    pub fn as_bytes(&self) -> &[u32; 4] {
        &self.0
    }

    /// Unpack from bytes received from a kernel.
    pub fn from_bytes(bytes: [u32; 4]) -> Self {
        Self(bytes)
    }
}

impl Default for TmaDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_new_is_zeroed() {
        let desc = TmaDescriptor::new();
        assert_eq!(desc.0, [0, 0, 0, 0]);
    }

    #[test]
    fn descriptor_gmem_addr() {
        let addr: u64 = 0x1234_5678;
        let desc = TmaDescriptor::new().with_gmem_addr(addr);
        assert_eq!(desc.gmem_addr(), addr);
    }

    #[test]
    fn descriptor_gmem_addr_max() {
        let addr: u64 = 0xFFFF_FFFF;
        let desc = TmaDescriptor::new().with_gmem_addr(addr);
        assert_eq!(desc.gmem_addr(), addr);
    }

    #[test]
    fn descriptor_gmem_addr_small() {
        let addr: u64 = 0x100;
        let desc = TmaDescriptor::new().with_gmem_addr(addr);
        assert_eq!(desc.gmem_addr(), addr);
    }

    #[test]
    fn descriptor_gmem_addr_truncates_upper_bits() {
        let addr: u64 = 0x1234_5678_9ABC_DEF0;
        let desc = TmaDescriptor::new().with_gmem_addr(addr);
        assert_eq!(desc.gmem_addr(), 0x9ABC_DEF0);
    }

    #[test]
    fn descriptor_with_box() {
        let desc = TmaDescriptor::new()
            .with_box(64, 128, 128, 256, 128, 128);

        // box X = 64 in word 1 [15:0]
        assert_eq!(desc.0[1] & 0xFFFF, 64u32);
        // gmem_x_stride = 128 in word 1 [23:16]
        assert_eq!((desc.0[1] >> 16) & 0xFF, 128u32);
        // smem_x_stride = 128 in word 1 [31:24]
        assert_eq!((desc.0[1] >> 24) & 0xFF, 128u32);
        // box Y = 256 in word 2 [15:0]
        assert_eq!(desc.0[2] & 0xFFFF, 256u32);
        // gmem_y_stride = 128 in word 2 [31:16]
        assert_eq!((desc.0[2] >> 16) & 0xFFFF, 128u32);
        // smem_y_stride = 128 in word 3 [31:16]
        assert_eq!((desc.0[3] >> 16) & 0xFFFF, 128u32);
    }

    #[test]
    fn descriptor_with_box_saturates_large_stride() {
        let desc = TmaDescriptor::new()
            .with_box(64, 500, 500, 256, 128, 128);

        assert_eq!((desc.0[1] >> 16) & 0xFF, 255u32);
        assert_eq!((desc.0[1] >> 24) & 0xFF, 255u32);
    }

    #[test]
    fn descriptor_with_element_info() {
        let desc = TmaDescriptor::new().with_element_info(1);
        assert_eq!((desc.0[1] >> 28) & 0xF, 1u32);
    }

    #[test]
    fn descriptor_with_descriptor_type() {
        let desc = TmaDescriptor::new().with_descriptor_type(1);
        assert_eq!((desc.0[3] >> 24) & 0xFF, 1u32);
    }

    #[test]
    fn descriptor_with_smem_config() {
        let desc = TmaDescriptor::new().with_smem_config(0);
        assert_eq!((desc.0[3] >> 28) & 0xF, 0u32);
    }

    #[test]
    fn descriptor_with_cache_hint() {
        let desc = TmaDescriptor::new().with_cache_hint(0);
        assert_eq!((desc.0[3] >> 30) & 0x3, 0u32);
    }

    #[test]
    fn descriptor_full_chain() {
        let addr: u64 = 0x2000;
        let desc = TmaDescriptor::new()
            .with_gmem_addr(addr)
            .with_box(64, 128, 128, 512, 128, 128)
            .with_element_info(1)
            .with_descriptor_type(1)
            .with_smem_config(0)
            .with_cache_hint(0);

        assert_eq!(desc.gmem_addr(), addr);
        assert_eq!(desc.0[1] & 0xFFFF, 64u32);
        assert_eq!(desc.0[2] & 0xFFFF, 512u32);
        assert_eq!((desc.0[3] >> 24) & 0xFF, 1u32);
    }

    #[test]
    fn descriptor_from_bytes_roundtrip() {
        let bytes = [1, 2, 3, 4];
        let desc = TmaDescriptor::from_bytes(bytes);
        assert_eq!(desc.0, bytes);
    }

    #[test]
    fn descriptor_clone_copy() {
        let desc = TmaDescriptor::new()
            .with_gmem_addr(0x1000)
            .with_box(64, 128, 128, 256, 128, 128);
        let dup = desc;
        assert_eq!(dup.gmem_addr(), desc.gmem_addr());
        assert_eq!(dup.0[1] & 0xFFFF, desc.0[1] & 0xFFFF);
        assert_eq!(dup.0[2], desc.0[2]);
    }
}
