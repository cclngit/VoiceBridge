; ==============================================================================
; VoiceBridge — Inno Setup 6 Windows Installer Script (x86_64 & ARM64)
; ==============================================================================

#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif

#ifndef MyArch
  #define MyArch "x64compatible"
#endif

#ifndef MyBinaryPath
  #define MyBinaryPath "..\..\target\x86_64-pc-windows-msvc\release\VoiceBridge.exe"
#endif

#ifndef MyOutputBaseFilename
  #define MyOutputBaseFilename "VoiceBridge-windows-x86_64-setup"
#endif

[Setup]
AppId={{8E92D64A-4B57-4E2A-9D8C-7A31E901FA24}
AppName=VoiceBridge
AppVersion={#MyAppVersion}
AppVerName=VoiceBridge {#MyAppVersion}
AppPublisher=VoiceBridge Contributors
AppPublisherURL=https://github.com/cclngit/VoiceBridge
AppSupportURL=https://github.com/cclngit/VoiceBridge
DefaultDirName={autopf}\VoiceBridge
DefaultGroupName=VoiceBridge
AllowNoIcons=yes
LicenseFile=..\..\LICENSE
OutputDir=..\..\dist
OutputBaseFilename={#MyOutputBaseFilename}
SetupIconFile=VoiceBridge.ico
UninstallDisplayIcon={app}\VoiceBridge.exe
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed={#MyArch}
ArchitecturesInstallIn64BitMode={#MyArch}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyBinaryPath}"; DestDir: "{app}"; DestName: "VoiceBridge.exe"; Flags: ignoreversion
#ifdef MyVulkanLoader
; Vulkan loader for the GPU build: the .exe cannot start without vulkan-1.dll. GPU drivers
; (NVIDIA / AMD / Intel) normally install it system-wide; ship it only when it is missing
; (VMs, Windows N, no GPU driver yet), so whisper.cpp starts and falls back to the CPU.
Source: "{#MyVulkanLoader}"; DestDir: "{app}"; DestName: "vulkan-1.dll"; Flags: ignoreversion; Check: not SystemHasVulkanLoader
#endif
#ifdef MyRuntimeDllDir
; Runtime DLLs the .exe imports, deployed next to it so nothing else has to be installed:
; - Visual C++ runtime (msvcp140, vcruntime140…; Microsoft app-local deployment);
; - x64: CUDA runtime for NVIDIA GPUs (cudart, cuBLAS, cuRAND — NVIDIA redistributables),
;   so Kyutai / MADLAD / Whisper run on CUDA with only the NVIDIA driver. Unused, and
;   harmless, on AMD / Intel PCs: those use Vulkan or the CPU.
Source: "{#MyRuntimeDllDir}\*.dll"; DestDir: "{app}"; Flags: ignoreversion
#endif
Source: "..\..\config.example.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\docs\logo.png"; DestDir: "{app}\docs"; Flags: ignoreversion
Source: "..\..\models\README.md"; DestDir: "{app}\models"; Flags: ignoreversion
Source: "..\..\scripts\asr_server.py"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\..\scripts\start_local_asr.bat"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\..\scripts\start_local_asr.sh"; DestDir: "{app}\scripts"; Flags: ignoreversion
Source: "..\..\scripts\README.md"; DestDir: "{app}\scripts"; Flags: ignoreversion

[Icons]
Name: "{group}\VoiceBridge"; Filename: "{app}\VoiceBridge.exe"
Name: "{group}\{cm:UninstallProgram,VoiceBridge}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\VoiceBridge"; Filename: "{app}\VoiceBridge.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VoiceBridge.exe"; Description: "{cm:LaunchProgram,VoiceBridge}"; Flags: nowait postinstall skipifsilent
; In-app update (silent install started by VoiceBridge itself): relaunch the new version.
Filename: "{app}\VoiceBridge.exe"; Flags: nowait skipifnotsilent


[Code]
function SystemHasVulkanLoader: Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\vulkan-1.dll'));
end;
