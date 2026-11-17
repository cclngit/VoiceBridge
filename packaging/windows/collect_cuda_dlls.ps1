# Copies the redistributable CUDA runtime DLLs that VoiceBridge.exe needs (cudart, cuBLAS,
# cuBLASLt, cuRAND, NVRTC…) from the CUDA Toolkit into -OutDir, following their own imports,
# so the installer runs on PCs without the CUDA Toolkit. Also checks that the NVIDIA driver
# library (nvcuda.dll, not redistributable) is only delay-loaded: otherwise the .exe would
# not start on AMD / Intel / driverless PCs.
#
# Requires dumpbin.exe on PATH (MSVC developer environment) and CUDA_PATH (or -CudaPath).
param(
  [Parameter(Mandatory = $true)][string]$Exe,
  [Parameter(Mandatory = $true)][string]$OutDir,
  [string]$CudaPath = $env:CUDA_PATH
)
$ErrorActionPreference = "Stop"

if (-not $CudaPath) { throw "CUDA_PATH is not set" }
$cudaBins = @("$CudaPath\bin", "$CudaPath\bin\x64") | Where-Object { Test-Path $_ }
if (-not $cudaBins) { throw "No bin directory under $CudaPath" }

# Returns @{ Normal = [...]; Delay = [...] } from `dumpbin /dependents`.
function Get-Imports([string]$Path) {
  $out = & dumpbin /nologo /dependents $Path
  if ($LASTEXITCODE -ne 0) { throw "dumpbin failed on $Path" }
  $normal = @(); $delay = @(); $section = $null
  foreach ($line in $out) {
    if ($line -match 'Image has the following dependencies') { $section = 'normal'; continue }
    if ($line -match 'Image has the following delay load dependencies') { $section = 'delay'; continue }
    if ($line -match '^\s*Summary') { $section = $null; continue }
    $name = $line.Trim()
    if ($section -and $name -match '\.dll$') {
      if ($section -eq 'normal') { $normal += $name } else { $delay += $name }
    }
  }
  return @{ Normal = $normal; Delay = $delay }
}

function Find-CudaDll([string]$Name) {
  foreach ($dir in $cudaBins) {
    $p = Join-Path $dir $Name
    if (Test-Path $p) { return $p }
  }
  return $null
}

$exeImports = Get-Imports $Exe
if ($exeImports.Normal -contains 'nvcuda.dll') {
  throw "$Exe imports nvcuda.dll directly: it must be delay-loaded (see build.rs)"
}
Write-Host "Delay-loaded: $($exeImports.Delay -join ', ')"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$queue = [System.Collections.Generic.Queue[string]]::new()
$exeImports.Normal + $exeImports.Delay | ForEach-Object { $queue.Enqueue($_) }
$copied = @{}
while ($queue.Count -gt 0) {
  $name = $queue.Dequeue()
  if ($copied.ContainsKey($name.ToLower())) { continue }
  $src = Find-CudaDll $name
  if (-not $src) { continue }
  Copy-Item $src (Join-Path $OutDir $name) -Force
  $copied[$name.ToLower()] = $src
  $deps = Get-Imports $src
  $deps.Normal + $deps.Delay | ForEach-Object { $queue.Enqueue($_) }
}

# NVRTC loads its builtins library dynamically (not visible to dumpbin).
if ($copied.Keys | Where-Object { $_ -like 'nvrtc64_*' }) {
  foreach ($dir in $cudaBins) {
    Get-ChildItem $dir -Filter 'nvrtc-builtins64_*.dll' | ForEach-Object {
      Copy-Item $_.FullName (Join-Path $OutDir $_.Name) -Force
      $copied[$_.Name.ToLower()] = $_.FullName
    }
  }
}

if ($copied.Count -eq 0) { throw "$Exe does not use any CUDA Toolkit DLL (was it built with --features cuda?)" }
foreach ($required in @('cudart64_*', 'cublas64_*', 'cublasLt64_*')) {
  if (-not ($copied.Keys | Where-Object { $_ -like $required.ToLower() })) {
    throw "Expected a $required DLL among the CUDA dependencies"
  }
}
Get-ChildItem $OutDir -Filter *.dll | ForEach-Object {
  Write-Host ("Bundled {0} ({1:N0} MB)" -f $_.Name, ($_.Length / 1MB))
}
