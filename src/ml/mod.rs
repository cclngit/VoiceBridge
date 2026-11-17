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
            if nvidia_driver_present() && candle_core::utils::cuda_is_available() {
                match Device::new_cuda(ordinal).and_then(|d| smoke_test(&d).map(|_| d)) {
                    Ok(d) => return d,
                    // e.g. a GPU older than the compute capability the kernels were built
                    // for (8.0 in release builds): the PTX cannot be loaded there.
                    Err(e) => tracing::warn!("CUDA device #{ordinal} unusable for Candle, using the CPU: {e}"),
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

/// Runs the kernels a model needs first (fill, matmul, cast) so an unusable GPU is detected
/// here, with a CPU fallback, instead of failing in the middle of model loading.
pub fn smoke_test(device: &Device) -> candle_core::Result<()> {
    let a = candle_core::Tensor::ones((8, 8), device.bf16_default_to_f32(), device)?;
    let sum = a.matmul(&a)?.to_dtype(candle_core::DType::F32)?.sum_all()?.to_scalar::<f32>()?;
    if sum != 512.0 {
        candle_core::bail!("wrong result {sum} for a test matmul");
    }
    Ok(())
}

/// Whether the NVIDIA driver API (`nvcuda.dll`) can be loaded.
///
/// On Windows the CUDA build delay-loads `nvcuda.dll` (see build.rs) so the .exe also starts
/// on PCs without an NVIDIA driver. Calling any CUDA driver function there would then abort
/// the process instead of returning an error, so it must be probed before Candle touches CUDA.
/// Elsewhere the library is linked normally and is necessarily present once the app runs.
#[cfg(feature = "cuda")]
pub fn nvidia_driver_present() -> bool {
    #[cfg(windows)]
    {
        use std::sync::OnceLock;
        static PRESENT: OnceLock<bool> = OnceLock::new();
        *PRESENT.get_or_init(|| {
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn LoadLibraryExW(name: *const u16, file: *mut std::ffi::c_void, flags: u32) -> *mut std::ffi::c_void;
            }
            // LOAD_LIBRARY_SEARCH_SYSTEM32: the driver always installs it in System32; never
            // pick up a stray copy from the app or current directory.
            const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x0000_0800;
            let name: Vec<u16> = "nvcuda.dll".encode_utf16().chain(std::iter::once(0)).collect();
            // The handle is kept on purpose: the delay-load helper reuses the loaded module.
            let handle = unsafe { LoadLibraryExW(name.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32) };
            let present = !handle.is_null();
            if !present {
                tracing::info!("No NVIDIA driver (nvcuda.dll) found: Candle models use the CPU");
            }
            present
        })
    }
    #[cfg(not(windows))]
    {
        true
    }
}

/// Human readable label of a Candle device.
pub fn device_label(device: &Device) -> &'static str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA GPU",
        Device::Metal(_) => "Metal GPU",
    }
}
