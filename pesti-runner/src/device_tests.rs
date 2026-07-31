//! Comprehensive tests for device.rs functionality.
//! Tests cover DeviceBackend, DeviceSelector, and hybrid routing logic.

#[cfg(test)]
mod device_backend_tests {
    use crate::device::{DeviceBackend, RunnerError};

    #[test]
    fn test_device_backend_new_defaults_to_cpu() {
        let backend = DeviceBackend::new("cuda");
        assert_eq!(backend.preference, "cuda");
        // Default device is CPU regardless of preference until select() is called
        match backend.device {
            candle_core::Device::Cpu => {}
            _ => panic!("Expected CPU device initially"),
        }
    }

    #[test]
    fn test_device_backend_select_cpu() {
        let mut backend = DeviceBackend::new("cpu");
        backend.select().unwrap();
        
        match backend.device {
            candle_core::Device::Cpu => {}
            _ => panic!("Expected CPU device after select(cpu)"),
        }
    }

    #[test]
    fn test_device_backend_select_mkl() {
        let mut backend = DeviceBackend::new("mkl");
        backend.select().unwrap();
        
        // MKL selects CPU backend
        match backend.device {
            candle_core::Device::Cpu => {}
            _ => panic!("Expected CPU device after select(mkl)"),
        }
    }

    #[test]
    fn test_device_backend_select_accelerate() {
        let mut backend = DeviceBackend::new("accelerate");
        backend.select().unwrap();
        
        // Accelerate selects CPU backend
        match backend.device {
            candle_core::Device::Cpu => {}
            _ => panic!("Expected CPU device after select(accelerate)"),
        }
    }

    #[test]
    fn test_device_backend_info_cpu() {
        let mut backend = DeviceBackend::new("cpu");
        backend.select().unwrap();
        
        let info = backend.info().unwrap();
        assert_eq!(info, "cpu");
    }

    #[test]
    fn test_device_backend_is_available_cpu() {
        let mut backend = DeviceBackend::new("cpu");
        backend.select().unwrap();
        
        let available = backend.is_available().unwrap();
        assert!(available, "CPU should always be available");
    }

    #[test]
    fn test_device_backend_unknown_preference_falls_back_to_cpu() {
        let mut backend = DeviceBackend::new("unknown_backend");
        backend.select().unwrap();
        
        // Unknown preference should fall back to CPU
        match backend.device {
            candle_core::Device::Cpu => {}
            _ => panic!("Expected CPU device for unknown preference"),
        }
    }

    #[test]
    fn test_device_backend_cuda_if_available() {
        let mut backend = DeviceBackend::new("cuda");
        backend.select().unwrap();
        
        // CUDA selection depends on hardware availability
        // Just verify it doesn't panic and selects something valid
        let _ = backend.info().unwrap();
        let _ = backend.is_available().unwrap();
    }
}

#[cfg(test)]
mod device_selector_tests {
    use super::super::*;
    use candle_core::Device;

    #[tokio::test]
    async fn test_device_selector_new_discovers_devices() {
        let selector = DeviceSelector::new();
        // Should have at least CPU in priority list
        assert!(selector.priority.contains(&DeviceType::Cpu) || !selector.priority.is_empty());
        // Should have discovered local devices
        assert!(!selector.local_devices.is_empty());
    }

    #[test]
    fn test_device_selector_with_explicit_priority() {
        let priority = vec![
            DeviceType::LocalGpu(0),
            DeviceType::LocalGpu(1),
            DeviceType::Cpu,
        ];
        let selector = DeviceSelector::with_priority(priority);
        assert_eq!(selector.priority.len(), 3);
        assert_eq!(selector.priority[0], DeviceType::LocalGpu(0));
    }

    #[tokio::test]
    async fn test_device_selector_refresh() {
        let mut selector = DeviceSelector::new();
        let initial_len = selector.local_devices.len();
        
        selector.refresh().await;
        
        // After refresh, should still have devices
        assert!(!selector.local_devices.is_empty());
    }

    #[tokio::test]
    async fn test_device_selector_select_for_model_small() {
        let mut selector = DeviceSelector::new();
        
        // Small model (100MB)
        let selection = selector.select_for_model(100 * 1024 * 1024).await;
        
        // Should select something (CPU at minimum)
        assert!(!selection.reason.is_empty());
        match &selection.device_type {
            DeviceType::LocalGpu(_) | DeviceType::Remote(_) | DeviceType::Cpu => {}
        }
    }

    #[tokio::test]
    async fn test_device_selector_select_for_model_large() {
        let mut selector = DeviceSelector::new();
        
        // Large model (30GB)
        let selection = selector.select_for_model(30 * 1024 * 1024 * 1024).await;
        
        // Should fall back to CPU or best available GPU
        assert!(!selection.reason.is_empty());
    }

    #[tokio::test]
    async fn test_device_selector_quick_select() {
        let selector = DeviceSelector::new();
        
        // Quick select without refresh
        let selection = selector.quick_select(500 * 1024 * 1024).await;
        
        assert!(!selection.reason.is_empty());
    }

    #[tokio::test]
    async fn test_device_selector_list_available() {
        let selector = DeviceSelector::new();
        let devices = selector.list_available();
        
        // Should always have at least CPU listed
        assert!(devices.iter().any(|d| d.name == "CPU"));
    }

    #[test]
    fn test_device_selector_list_shows_cpu() {
        let selector = DeviceSelector::new();
        let devices = selector.list_available();
        
        let cpu_device = devices.iter().find(|d| d.name == "CPU").unwrap();
        assert_eq!(cpu_device.device_type, "cpu");
        assert!(cpu_device.available);
    }
}

#[cfg(test)]
mod device_type_tests {
    use super::super::*;

    #[test]
    fn test_device_type_local_gpu() {
        let gpu = DeviceType::LocalGpu(0);
        let json = serde_json::to_string(&gpu).unwrap();
        assert!(json.contains("LocalGpu"));
        assert!(json.contains("0"));
    }

    #[test]
    fn test_device_type_remote() {
        let remote = DeviceType::Remote("http://localhost:8080".to_string());
        let json = serde_json::to_string(&remote).unwrap();
        assert!(json.contains("Remote"));
        assert!(json.contains("localhost"));
    }

    #[test]
    fn test_device_type_cpu() {
        let cpu = DeviceType::Cpu;
        let json = serde_json::to_string(&cpu).unwrap();
        assert!(json.contains("Cpu"));
    }

    #[test]
    fn test_device_type_equality() {
        let gpu0 = DeviceType::LocalGpu(0);
        let gpu0_copy = DeviceType::LocalGpu(0);
        let gpu1 = DeviceType::LocalGpu(1);
        
        assert_eq!(gpu0, gpu0_copy);
        assert_ne!(gpu0, gpu1);
    }
}

#[cfg(test)]
mod device_selection_tests {
    use super::super::*;

    #[test]
    fn test_device_selection_is_remote_local() {
        let selection = DeviceSelection {
            device_type: DeviceType::LocalGpu(0),
            selected: LocalDevice::cpu_fallback(),
            remote: None,
            reason: "test".to_string(),
        };
        
        assert!(!selection.is_remote());
        assert!(selection.remote_endpoint().is_none());
    }

    #[test]
    fn test_device_selection_is_remote_true() {
        let remote = RemoteDevice {
            name: "Remote GPU".to_string(),
            endpoint: "http://192.168.1.100:8080".to_string(),
            healthy: true,
            latency_ms: 50,
            vram_total: 8_000_000_000,
            vram_free: 4_000_000_000,
        };
        
        let selection = DeviceSelection {
            device_type: DeviceType::Remote("test".to_string()),
            selected: LocalDevice::cpu_fallback(),
            remote: Some(remote),
            reason: "remote inference".to_string(),
        };
        
        assert!(selection.is_remote());
        assert_eq!(
            selection.remote_endpoint(),
            Some("http://192.168.1.100:8080")
        );
    }

    #[test]
    fn test_device_selection_cpu_fallback() {
        let selection = DeviceSelection {
            device_type: DeviceType::Cpu,
            selected: LocalDevice::cpu_fallback(),
            remote: None,
            reason: "CPU fallback".to_string(),
        };
        
        assert!(!selection.is_remote());
        assert_eq!(selection.device_type, DeviceType::Cpu);
    }
}

#[cfg(test)]
mod local_device_tests {
    use super::super::*;

    #[test]
    fn test_cpu_fallback_device() {
        let cpu = LocalDevice::cpu_fallback();
        
        assert_eq!(cpu.ordinal, u32::MAX);
        assert!(cpu.name.contains("CPU"));
        assert_eq!(cpu.total_vram, 0);
        assert_eq!(cpu.free_vram, 0);
        assert!(cpu.available);
    }

    #[test]
    fn test_local_device_can_hold_model() {
        let device = LocalDevice {
            ordinal: 0,
            name: "Test GPU".to_string(),
            total_vram: 16_000_000_000,
            free_vram: 8_000_000_000,
            compute_capability: "8.6".to_string(),
            available: true,
            used_vram: 2_000_000_000,
        };
        
        // Model that fits
        assert!(device.can_hold_model(5_000_000_000));
        
        // Model too large
        assert!(!device.can_hold_model(10_000_000_000));
    }

    #[test]
    fn test_local_device_zero_free_vram() {
        let device = LocalDevice {
            ordinal: 0,
            name: "Full GPU".to_string(),
            total_vram: 16_000_000_000,
            free_vram: 0,
            compute_capability: "8.6".to_string(),
            available: true,
            used_vram: 16_000_000_000,
        };
        
        // Even tiny model won't fit
        assert!(!device.can_hold_model(1));
    }
}

#[cfg(test)]
mod device_info_tests {
    use super::super::*;

    #[test]
    fn test_device_info_serialization() {
        let info = DeviceInfo {
            name: "RTX 4070".to_string(),
            device_type: "local_gpu".to_string(),
            vram_total: Some(16_000_000_000),
            vram_free: Some(8_000_000_000),
            available: true,
            ordinal: Some(0),
            endpoint: None,
        };
        
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("RTX 4070"));
        assert!(json.contains("16000000000"));
    }

    #[test]
    fn test_device_info_remote_serialization() {
        let info = DeviceInfo {
            name: "Remote LM Studio".to_string(),
            device_type: "remote".to_string(),
            vram_total: Some(8_000_000_000),
            vram_free: Some(4_000_000_000),
            available: true,
            ordinal: None,
            endpoint: Some("http://192.168.1.100:8080".to_string()),
        };
        
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("Remote LM Studio"));
        assert!(json.contains("endpoint"));
    }
}
