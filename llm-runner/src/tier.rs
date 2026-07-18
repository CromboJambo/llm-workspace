//! Tiered execution infrastructure for profile-driven optimization paths.
//!
//! Inspired by lumen's tier-0→tier-1→tier-2 model, but optimized for GPU-first workflows:
//! - **Tier 1 (CPU baseline)**: Pure-Rust transformer kernels (correctness oracle)
//! - **Tier 2 (GPU flash attention)**: cuda-oxide WGMMA + tcgen05 optimizations
//! - **Tier 3 (llama.cpp FFI CUDA)**: Full model offload to optimized backend

use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::debug;

/// Execution tier with profile-driven tier-up thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// CPU baseline — generic kernels, always available
    CpuBaseline = 0,
    /// GPU flash attention — cuda-oxide WGMMA kernels, requires VRAM
    GpuFlashAttention = 1,
    /// Full CUDA backend — llama.cpp FFI with optimized kernels
    GpuFullBackend = 2,
}

impl Tier {
    pub fn name(&self) -> &'static str {
        match self {
            Self::CpuBaseline => "cpu_baseline",
            Self::GpuFlashAttention => "gpu_flash_attention",
            Self::GpuFullBackend => "gpu_full_backend",
        }
    }

    /// Is this tier GPU-accelerated?
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::GpuFlashAttention | Self::GpuFullBackend)
    }

    /// Get the default tier-up threshold (invocations before tiering up).
    pub fn tier_up_threshold(&self) -> usize {
        match self {
            Self::CpuBaseline => 100,      // Tier up after 100 invocations
            Self::GpuFlashAttention => 500, // Stay at GPU attention until 500
            Self::GpuFullBackend => usize::MAX, // Never tier up (already max)
        }
    }

    /// Convert from numeric ID.
    pub fn from_id(id: u32) -> Tier {
        match id {
            0 => Tier::CpuBaseline,
            1 => Tier::GpuFlashAttention,
            _ => Tier::GpuFullBackend,
        }
    }

    /// Get numeric ID.
    pub fn as_u32(&self) -> u32 {
        *self as u32
    }

    /// Get tier-up threshold by numeric ID (for helper functions).
    fn threshold_by_id(id: u32) -> usize {
        match id {
            0 => 100,      // CPU baseline → GPU flash attention
            1 => 500,      // GPU flash attention → full backend
            _ => usize::MAX, // Already at max tier
        }
    }
}

/// Profile-driven execution state with call-count tracking.
pub struct TieredExecution {
    current_tier: AtomicUsize, // Numeric ID (u32 cast to usize for atomic ops)
    call_count: AtomicUsize,   // Cumulative invocations per tier
    tier_up_threshold: usize,  // Threshold for tier-up decision
}

impl TieredExecution {
    pub fn new(initial_tier: Tier) -> Self {
        Self {
            current_tier: AtomicUsize::new(initial_tier.as_u32() as usize),
            call_count: AtomicUsize::new(0),
            tier_up_threshold: initial_tier.tier_up_threshold(),
        }
    }

    /// Get the current execution tier.
    pub fn current_tier(&self) -> Tier {
        let val = self.current_tier.load(Ordering::Relaxed);
        Tier::from_id(val as u32)
    }

    /// Record a layer invocation and check if tier-up is needed.
    pub fn record_invocation(&self) -> Option<Tier> {
        let count = self.call_count.fetch_add(1, Ordering::Relaxed) + 1;

        // Check if we should tier up based on threshold for current tier
        let current_tier_id = self.current_tier.load(Ordering::Relaxed);
        let threshold = Tier::threshold_by_id(current_tier_id as u32);

        debug!(
            invocation_count = count,
            threshold = threshold,
            "Tiered execution: recorded invocation"
        );

        if count >= threshold {
            // Tier up logic (simplified for Phase 5.3 MVP)
            let new_tier_id = current_tier_id + 1;

            // Reset call counter on tier transition
            self.call_count.store(0, Ordering::Relaxed);
            self.current_tier.store(new_tier_id, Ordering::Relaxed);

            debug!(
                from_tier = ?Tier::from_id(current_tier_id as u32),
                to_tier = ?Tier::from_id(new_tier_id as u32),
                "Tiered execution: tier-up triggered"
            );

            Some(Tier::from_id(new_tier_id as u32))
        } else {
            None
        }
    }

    /// Manually set the current tier (for testing or forced switching).
    pub fn set_tier(&self, tier: Tier) {
        self.current_tier
            .store(tier.as_u32() as usize, Ordering::Relaxed);
        self.call_count.store(0, Ordering::Relaxed); // Reset counter on tier change
    }

    /// Get the invocation count for current tier.
    pub fn invocation_count(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }

    /// Check if GPU execution is available (for tier-up decision).
    pub fn has_gpu_available(&self) -> bool {
        // Phase 5.3 MVP: assume GPU always available
        // TODO: integrate with DeviceSelector for real VRAM checks
        true
    }

    /// Reset profile counters (e.g., after model unload or session timeout).
    pub fn reset_profile(&self) {
        self.call_count.store(0, Ordering::Relaxed);
    }
}

/// Layer-level profiling hook for tier-up decisions.
pub struct LayerProfiler {
    layer_id: String,
    call_count: AtomicUsize,
}

impl LayerProfiler {
    pub fn new(layer_name: &str) -> Self {
        Self {
            layer_id: layer_name.to_string(),
            call_count: AtomicUsize::new(0),
        }
    }

    /// Record a forward pass invocation for this layer.
    pub fn record_forward(&self) -> usize {
        let count = self.call_count.fetch_add(1, Ordering::Relaxed) + 1;

        if count % 50 == 0 { // Log every 50 invocations
            debug!(layer = %self.layer_id, count = count, "Layer profiler: invocation recorded");
        }

        count
    }

    /// Get current invocation count.
    pub fn count(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_encoding() {
        assert_eq!(Tier::from_id(0), Tier::CpuBaseline);
        assert_eq!(Tier::from_id(1), Tier::GpuFlashAttention);
        assert_eq!(Tier::from_id(2), Tier::GpuFullBackend);

        assert_eq!(Tier::CpuBaseline.as_u32(), 0);
        assert_eq!(Tier::GpuFlashAttention.as_u32(), 1);
    }

    #[test]
    fn test_tier_up_threshold() {
        assert_eq!(Tier::threshold_by_id(0), 100); // CPU → GPU flash attention
        assert_eq!(Tier::threshold_by_id(1), 500); // Flash → full backend
        assert_eq!(Tier::threshold_by_id(2), usize::MAX); // Already max tier
    }

    #[test]
    fn test_tiered_execution_invocation_tracking() {
        let execution = TieredExecution::new(Tier::CpuBaseline);

        // First 99 invocations should not trigger tier-up
        for _ in 0..99 {
            assert_eq!(execution.current_tier(), Tier::CpuBaseline);
            assert!(execution.record_invocation().is_none());
        }

        // 100th invocation triggers tier-up to GPU flash attention
        let tier_up = execution.record_invocation().unwrap();
        assert_eq!(tier_up, Tier::GpuFlashAttention);
    }

    #[test]
    fn test_layer_profiler() {
        let profiler = LayerProfiler::new("transformer.layer.0");

        assert_eq!(profiler.count(), 0);

        for i in 1..=51 {
            let count = profiler.record_forward();
            if i % 50 == 0 {
                // Should log at multiples of 50
                assert_eq!(count, i);
            }
        }

        assert_eq!(profiler.count(), 51);
    }

    #[test]
    fn test_tiered_execution_reset() {
        let execution = TieredExecution::new(Tier::CpuBaseline);

        for _ in 0..150 {
            execution.record_invocation();
        }

        assert!(execution.current_tier().is_gpu()); // Should have tiered up

        execution.reset_profile();
        assert_eq!(execution.invocation_count(), 0);
    }

    #[test]
    fn test_manual_tier_switch() {
        let execution = TieredExecution::new(Tier::CpuBaseline);

        for _ in 0..25 {
            execution.record_invocation();
        }

        assert_eq!(execution.current_tier(), Tier::CpuBaseline); // Still at CPU

        execution.set_tier(Tier::GpuFullBackend);
        assert_eq!(execution.current_tier(), Tier::GpuFullBackend);
        assert_eq!(execution.invocation_count(), 0); // Counter reset
    }

    #[test]
    fn test_gpu_detection() {
        let execution = TieredExecution::new(Tier::CpuBaseline);

        assert!(execution.has_gpu_available()); // MVP assumes GPU always available

        // Can tier up if GPU is available
        for _ in 0..150 {
            execution.record_invocation();
        }

        assert!(execution.current_tier().is_gpu());
    }

    #[test]
    fn test_tier_name() {
        assert_eq!(Tier::CpuBaseline.name(), "cpu_baseline");
        assert_eq!(Tier::GpuFlashAttention.name(), "gpu_flash_attention");
        assert_eq!(Tier::GpuFullBackend.name(), "gpu_full_backend");
    }

    #[test]
    fn test_is_gpu() {
        assert!(!Tier::CpuBaseline.is_gpu());
        assert!(Tier::GpuFlashAttention.is_gpu());
        assert!(Tier::GpuFullBackend.is_gpu());
    }

    #[test]
    fn tiered_execution_threshold() {
        let execution = TieredExecution::new(Tier::CpuBaseline);

        // Check that thresholds are set correctly based on initial tier
        match Tier::CpuBaseline.tier_up_threshold() {
            100 => {} // CPU baseline threshold
            _ => panic!("Expected 100"),
        }
    }

    #[test]
    fn test_tier_from_id_unknown() {
        // Unknown IDs should default to full backend (max tier)
        assert_eq!(Tier::from_id(99), Tier::GpuFullBackend);
        assert_eq!(Tier::from_id(u32::MAX), Tier::GpuFullBackend);
    }

    #[test]
    fn test_single_tier_transition() {
        let execution = TieredExecution::new(Tier::CpuBaseline);

        // First tier-up happens at 100 invocations
        for _ in 0..99 {
            assert!(execution.record_invocation().is_none());
        }

        // Should switch to GPU flash attention
        let new_tier = execution.record_invocation().unwrap();
        assert_eq!(new_tier, Tier::GpuFlashAttention);

        // After tier-up, counter was reset; verify we stay at this tier for 250 more invocations
        for _ in 0..250 {
            assert!(execution.record_invocation().is_none());
        }

        assert_eq!(execution.current_tier(), Tier::GpuFlashAttention);
    }
}
