; Inno Setup Script for Voxel Headless Server
; https://github.com/larvance/telos

#ifndef MyAppVersion
#define MyAppVersion "0.1.0"
#endif

#define MyAppName "Voxel Server"
#define MyAppPublisher "Larvance"
#define MyAppURL "https://github.com/larvance/telos"
#define MyAppExeName "voxel-server.exe"

[Setup]
AppId={{C7A92F4D-3E89-4D6C-8B12-0A5B5F3E1B9B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
DefaultGroupName={#MyAppName}
OutputDir=..\..\target\dist
OutputBaseFilename=voxel-server-{#MyAppVersion}-windows-x86_64-setup
SetupIconFile=voxel.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "..\..\target\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "voxel.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\server.toml"; DestDir: "{app}"; Flags: ignoreversion onlyifdoesntexist

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\voxel.ico"
