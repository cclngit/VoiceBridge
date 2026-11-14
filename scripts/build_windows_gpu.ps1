<#
.SYNOPSIS
  Build VoiceBridge on Windows with NVIDIA CUDA (default) or Vulkan GPU acceleration.

.DESCRIPTION
  On macOS, Metal is enabled automatically. On Windows the GPU backends are opt-in
  Cargo features, and nvcc needs the MSVC toolchain (cl.exe) on PATH. This script
  loads the Visual Studio x64 environment, points at the CUDA Toolkit and libclang,
  then runs cargo.

.EXAMPLE
  .\scripts\build_windows_gpu.ps1                 # cargo build --release --features cuda
  .\scripts\build_windows_gpu.ps1 -Run            # build, then run
  .\scripts\build_windows_gpu.ps1 -Features vulkan
  .\scripts\build_windows_gpu.ps1 -ComputeCap 89  # e.g. RTX 40xx (auto-detected by default)
#>
param(
    [string]$Features = "cuda",
    [string]$ComputeCap = "",
    [switch]$Run,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

# 1. MSVC environment (cl.exe, link.exe, Windows SDK)
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found: install Visual Studio Build Tools (C++ workload)." }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "No Visual Studio installation with the C++ x64 toolset was found." }
$env:PATH = "$(Split-Path $vswhere);$env:PATH"
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw "cl.exe still not on PATH after vcvars64.bat." }

# 2. libclang for bindgen (whisper-rs-sys)
if (-not $env:LIBCLANG_PATH) {
    foreach ($d in @("C:\Program Files\LLVM\bin", "$vs\VC\Tools\Llvm\x64\bin")) {
        if (Test-Path "$d\libclang.dll") { $env:LIBCLANG_PATH = $d; break }
    }
}
if (-not $env:LIBCLANG_PATH) { throw "libclang.dll not found: winget install LLVM.LLVM" }

# 3. CUDA Toolkit + target GPU architecture
if ($Features -match "cuda") {
    # Pull CUDA_PATH and CUDA_PATH_Vxx_y from the machine env: the MSBuild CUDA targets used by
    # whisper.cpp's CMake need CUDA_PATH_Vxx_y, which a shell opened before the install won't have.
    $machineEnv = [Environment]::GetEnvironmentVariables("Machine")
    foreach ($k in $machineEnv.Keys) {
        if ($k -like "CUDA_PATH*" -and -not (Test-Path "env:$k")) { Set-Item -Path "env:$k" -Value $machineEnv[$k] }
    }
    if (-not $env:CUDA_PATH -or -not (Test-Path "$env:CUDA_PATH\bin\nvcc.exe")) {
        throw "CUDA Toolkit not found: winget install Nvidia.CUDA -v 12.9"
    }
    $env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
    # nvcc defaults to the static CRT (/MT) on Windows, so candle-kernels' libmoe.a would clash
    # with whisper.cpp (/MD) at link time (LNK2038 RuntimeLibrary mismatch). Prepended so that
    # explicit flags (e.g. CMake's) still win.
    $env:NVCC_PREPEND_FLAGS = "-Xcompiler=/MD"
    if (-not $ComputeCap) {
        $cap = (& nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
        if ($cap) { $ComputeCap = $cap.Trim().Replace(".", "") }
    }
    if ($ComputeCap) {
        $env:CUDA_COMPUTE_CAP = $ComputeCap          # candle-kernels
        $env:CMAKE_CUDA_ARCHITECTURES = $ComputeCap  # whisper.cpp / ggml
        Write-Host "CUDA compute capability: sm_$ComputeCap"
    }
}

# 4. Build / run
[string[]]$cargoArgs = @(if ($Run) { "run" } else { "build" })
if (-not $Dev) { $cargoArgs += "--release" }
$cargoArgs += @("--features", $Features)
Write-Host "cargo $($cargoArgs -join ' ')"
& cargo @cargoArgs
exit $LASTEXITCODE
