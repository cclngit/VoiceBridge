//! Shared machine-learning helpers for the in-process Candle models
//! (Kyutai STT, Hibiki, MADLAD-400).

pub mod spm;

use candle_core::Device;

/// Which model a device instance belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceSlot {
    Streaming,
    Translation,
}

/// Resolve the Candle device used by Kyutai / Hibiki / MADLAD.
///
/// Metal is used on macOS and CUDA when the `cuda` feature is compiled in. Other GPU
/// backends (Vulkan / ROCm) are only available to whisper.cpp, so Candle models fall
/// back to the CPU there.
///
/// Each model gets its own cached device instance (`slot`): Candle tensors may only be
/// combined with tensors of the very same instance, and a Candle Metal device shares one
/// command buffer between threads — models running concurrently (e.g. Kyutai streaming
/// while MADLAD translates) on a shared instance produce corrupted results.
pub fn resolve_device(slot: DeviceSlot, use_gpu: bool, gpu_device: i32) -> Device {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    type DeviceCache = Mutex<HashMap<(DeviceSlot, bool, i32), Device>>;
    static CACHE: OnceLock<DeviceCache> = OnceLock::new();
    let key = (slot, use_gpu, gpu_device.max(0));
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(map) = cache.lock()
        && let Some(d) = map.get(&key)
    {
        return d.clone();
    }
    let device = create_device(use_gpu, gpu_device);
    if let Ok(mut map) = cache.lock() {
        map.insert(key, device.clone());
    }
    device
}

fn create_device(use_gpu: bool, gpu_device: i32) -> Device {
    let ordinal = gpu_device.max(0) as usize;
    if use_gpu {
        #[cfg(feature = "cuda")]
        {
            if candle_core::utils::cuda_is_available() {
                match Device::new_cuda(ordinal) {
                    Ok(d) => return d,
                    Err(e) => tracing::warn!("CUDA device #{ordinal} unavailable for Candle: {e}"),
                }
            }
        }
        #[cfg(any(target_os = "macos", feature = "metal"))]
        {
            if candle_core::utils::metal_is_available() {
                match Device::new_metal(ordinal) {
                    Ok(d) => return d,
                    Err(e) => tracing::warn!("Metal device #{ordinal} unavailable for Candle: {e}"),
                }
            }
        }
    }
    let _ = ordinal;
    Device::Cpu
}

/// Human readable label of a Candle device.
pub fn device_label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA GPU",
        Device::Metal(_) => "Metal GPU",
    }
}
