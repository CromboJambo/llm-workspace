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

/// 128-bit TMA descriptor packed as u128 for correct alignment and zero-copy casting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct TmaDescriptor(pub u128);

impl TmaDescriptor {
    /// Create a new zeroed descriptor.
    pub const fn new() -> Self {
        Self(0u128)
    }

    /// Get the underlying u128 value.
    pub const fn as_u128(&self) -> u128 {
        self.0
    }

    /// Get the descriptor as [u32; 4] for hardware register access.
    pub const fn as_u32_words(&self) -> [u32; 4] {
        [
            (self.0) as u32,
            (self.0 >> 32) as u32,
            (self.0 >> 64) as u32,
            (self.0 >> 96) as u32,
        ]
    }

    /// Set the GMEM address offset (within the buffer base).
    ///
    /// The offset must be 256-byte aligned for TMA.
    /// Stored in word 0 [31:0].
    pub const fn with_gmem_addr(mut self, addr: u64) -> Self {
        self.0 |= (addr as u128 & 0xFFFFFFFF) << 0;
        self
    }

    /// Get the GMEM address offset from the descriptor.
    pub const fn gmem_addr(&self) -> u64 {
        (self.0 & 0xFFFFFFFF) as u64
    }

    /// Set the box (region) dimensions and strides.
    ///
    /// `box_x` — number of elements along X axis (16-bit).
    /// `gmem_x_stride` — GMEM stride in elements between consecutive rows (8-bit, max 255).
    /// `smem_x_stride` — SMEM stride in elements between consecutive rows (8-bit, max 255).
    /// `box_y` — number of elements along Y axis (16-bit).
    /// `gmem_y_stride` — GMEM stride in elements between consecutive columns (16-bit).
    /// `smem_y_stride` — SMEM stride in elements between consecutive columns (16-bit).
    #[allow(clippy::identity_op, clippy::erasing_op)]
    pub const fn with_box(
        mut self,
        box_x: u16,
        gmem_x_stride: u16,
        smem_x_stride: u16,
        box_y: u16,
        gmem_y_stride: u16,
        smem_y_stride: u16,
    ) -> Self {
        self.0 |= (box_x as u128) << 32;
        let gmem_x = if gmem_x_stride > 255 { 255u16 } else { gmem_x_stride };
        let smem_x = if smem_x_stride > 255 { 255u16 } else { smem_x_stride };
        self.0 |= (gmem_x as u128) << 48;
        self.0 |= (smem_x as u128) << 56;
        self.0 |= (box_y as u128) << 64;
        self.0 |= (gmem_y_stride as u128) << 80;
        self.0 |= (smem_y_stride as u128) << 96;
        self
    }

    /// Set the element info field.
    pub const fn with_element_info(mut self, element_size: u8) -> Self {
        self.0 |= (element_size as u128 & 0xF) << 112;
        self
    }

    /// Set the descriptor type.
    pub const fn with_descriptor_type(mut self, dtype: u8) -> Self {
        self.0 |= (dtype as u128) << 112;
        self
    }

    /// Set the SMEM config field.
    pub const fn with_smem_config(mut self, config: u8) -> Self {
        self.0 |= (config as u128) << 128;
        self
    }

    /// Set the cache hint.
    pub const fn with_cache_hint(mut self, hint: u8) -> Self {
        self.0 |= (hint as u128 & 0x3) << 130;
        self
    }

    /// Unpack from [u32; 4] words received from a kernel.
    pub const fn from_u32_words(words: [u32; 4]) -> Self {
        Self(
            (words[0] as u128)
                | ((words[1] as u128) << 32)
                | ((words[2] as u128) << 64)
                | ((words[3] as u128) << 96),
        )
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
        assert_eq!(desc.as_u128(), 0);
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
        let desc = TmaDescriptor::new().with_box(64, 128, 128, 256, 128, 128);
        let words = desc.as_u32_words();

        assert_eq!(words[0] & 0xFFFFFFFF, 0u32);
        assert_eq!(words[1] & 0xFFFF, 64u32);
        assert_eq!((words[1] >> 16) & 0xFF, 128u32);
        assert_eq!((words[1] >> 24) & 0xFF, 128u32);
        assert_eq!(words[2] & 0xFFFF, 256u32);
        assert_eq!((words[2] >> 16) & 0xFFFF, 128u32);
    }

    #[test]
    fn descriptor_with_box_saturates_large_stride() {
        let desc = TmaDescriptor::new().with_box(64, 500, 500, 256, 128, 128);
        let words = desc.as_u32_words();

        assert_eq!((words[0] >> 16) & 0xFF, 255u32);
        assert_eq!((words[0] >> 24) & 0xFF, 255u32);
    }

    #[test]
    fn descriptor_with_element_info() {
        let desc = TmaDescriptor::new().with_element_info(1);
        let words = desc.as_u32_words();
        assert_eq!((words[3] >> 16) & 0xF, 1u32);
    }

    #[test]
    fn descriptor_with_descriptor_type() {
        let desc = TmaDescriptor::new().with_descriptor_type(1);
        let words = desc.as_u32_words();
        assert_eq!((words[2] >> 8) & 0xFF, 1u32);
    }

    #[test]
    fn descriptor_with_smem_config() {
        let desc = TmaDescriptor::new().with_smem_config(0);
        let words = desc.as_u32_words();
        assert_eq!((words[3] >> 8) & 0xF, 0u32);
    }

    #[test]
    fn descriptor_with_cache_hint() {
        let desc = TmaDescriptor::new().with_cache_hint(0);
        let words = desc.as_u32_words();
        assert_eq!((words[3] >> 26) & 0x3, 0u32);
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
        let words = desc.as_u32_words();
        assert_eq!(words[0] & 0xFFFF, 64u32);
        assert_eq!(words[1] & 0xFFFF, 512u32);
        assert_eq!((words[2] >> 8) & 0xFF, 1u32);
    }

    #[test]
    fn descriptor_from_words_roundtrip() {
        let words = [1, 2, 3, 4];
        let desc = TmaDescriptor::from_u32_words(words);
        assert_eq!(desc.as_u32_words(), words);
    }

    #[test]
    fn descriptor_clone_copy() {
        let desc = TmaDescriptor::new()
            .with_gmem_addr(0x1000)
            .with_box(64, 128, 128, 256, 128, 128);
        let dup = desc;
        assert_eq!(dup.gmem_addr(), desc.gmem_addr());
        assert_eq!(dup.as_u32_words()[0] & 0xFFFF, desc.as_u32_words()[0] & 0xFFFF);
        assert_eq!(dup.as_u32_words()[1], desc.as_u32_words()[1]);
    }
}
