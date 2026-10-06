; Inno Setup Script for Voxel Game Client
; https://github.com/larvance/voxel

#ifndef MyAppVersion
#define MyAppVersion "0.1.0"
#endif

#define MyAppName "Voxel"
#define MyAppPublisher "Larvance"
#define MyAppURL "https://github.com/larvance/voxel"
#define MyAppExeName "voxel.exe"

[Setup]
AppId={{E1B93F7B-2E89-4D6C-9A52-0A5B5F3E1B9A}
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
OutputBaseFilename=voxel-{#MyAppVersion}-windows-x86_64-setup
SetupIconFile=voxel.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\..\target\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "voxel.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\assets\voxel\*"; DestDir: "{app}\assets\voxel"; Flags: ignoreversion recursesubdirs createallsubdirs; Flags: skipifsourcedoesntexist

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\voxel.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\voxel.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
