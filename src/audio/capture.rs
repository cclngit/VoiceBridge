use super::resampler::{calculate_rms, AudioConverter};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc::UnboundedSender;

pub struct AudioDeviceManager;

#[derive(Debug, Clone, serde::Serialize)]
pub struct DeviceInfo {
    pub name: String,
    pub is_default: bool,
}

impl AudioDeviceManager {
    pub fn list_input_devices() -> Vec<DeviceInfo> {
        let host = cpal::default_host();
        let default_device_name = host.default_input_device().map(|d| d.to_string());

        let mut devices = Vec::new();
        if let Ok(input_devices) = host.input_devices() {
            for device in input_devices {
                let name = device.to_string();
                let is_default = default_device_name.as_ref() == Some(&name);
                devices.push(DeviceInfo { name, is_default });
            }
        }
        devices
    }
}

pub struct AudioCapture {
    _stream: cpal::Stream,
    is_running: Arc<AtomicBool>,
}

impl AudioCapture {
    pub fn start(
        device_name: Option<&str>,
        target_sample_rate: u32,
        audio_tx: UnboundedSender<Vec<f32>>,
        rms_tx: Option<tokio::sync::watch::Sender<f32>>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let host = cpal::default_host();
        let device = match device_name.map(str::trim).filter(|n| !n.is_empty()) {
            Some(name) => match host.input_devices()?.find(|d| d.to_string() == name) {
                Some(found) => found,
                None => {
                    tracing::warn!(
                        "Configured audio device '{}' not found on this system; falling back to default input device.",
                        name
                    );
                    host.default_input_device()
                        .ok_or_else(|| "No default audio input device found")?
                }
            },
            None => host
                .default_input_device()
                .ok_or_else(|| "No default audio input device found")?,
        };

        let device_name = device.to_string();
        tracing::info!("Using audio input device: {}", device_name);

        let default_config = device.default_input_config()?;
        let sample_format = default_config.sample_format();
        let config: StreamConfig = default_config.into();

        let source_rate = config.sample_rate;
        let channels = config.channels as usize;
        tracing::info!(
            "Hardware stream format: {} Hz, {} channels, format: {:?}",
            source_rate,
            channels,
            sample_format
        );

        let converter = Arc::new(AudioConverter::new(
            source_rate,
            target_sample_rate,
            channels,
        ));
        let is_running = Arc::new(AtomicBool::new(true));
        let running_clone = is_running.clone();

        let err_fn = |err| {
            tracing::error!("CPAL audio input stream error: {}", err);
        };

        let stream = match sample_format {
            SampleFormat::F32 => {
                let conv = converter.clone();
                let tx = audio_tx.clone();
                let rtx = rms_tx.clone();
                let is_run = running_clone.clone();

                device.build_input_stream(
                    config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        if !is_run.load(Ordering::Relaxed) {
                            return;
                        }
                        let resampled = conv.process(data);
                        if let Some(ref rtx) = rtx {
                            let rms = calculate_rms(&resampled);
                            let _ = rtx.send(rms);
                        }
                        let _ = tx.send(resampled);
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::I16 => {
                let conv = converter.clone();
                let tx = audio_tx.clone();
                let rtx = rms_tx.clone();
                let is_run = running_clone.clone();

                device.build_input_stream(
                    config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        if !is_run.load(Ordering::Relaxed) {
                            return;
                        }
                        let float_data: Vec<f32> =
                            data.iter().map(|&s| s as f32 / 32768.0).collect();
                        let resampled = conv.process(&float_data);
                        if let Some(ref rtx) = rtx {
                            let rms = calculate_rms(&resampled);
                            let _ = rtx.send(rms);
                        }
                        let _ = tx.send(resampled);
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::U16 => {
                let conv = converter.clone();
                let tx = audio_tx.clone();
                let rtx = rms_tx.clone();
                let is_run = running_clone.clone();

                device.build_input_stream(
                    config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        if !is_run.load(Ordering::Relaxed) {
                            return;
                        }
                        let float_data: Vec<f32> = data
                            .iter()
                            .map(|&s| (s as f32 - 32768.0) / 32768.0)
                            .collect();
                        let resampled = conv.process(&float_data);
                        if let Some(ref rtx) = rtx {
                            let rms = calculate_rms(&resampled);
                            let _ = rtx.send(rms);
                        }
                        let _ = tx.send(resampled);
                    },
                    err_fn,
                    None,
                )?
            }
            sample_format => {
                return Err(format!("Unsupported sample format: {:?}", sample_format).into());
            }
        };

        stream.play()?;
        tracing::info!("Audio capture stream started successfully.");

        Ok(Self {
            _stream: stream,
            is_running,
        })
    }

    pub fn set_active(&self, active: bool) {
        self.is_running.store(active, Ordering::Relaxed);
    }

    #[allow(dead_code)]
    pub fn is_active(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }
}
