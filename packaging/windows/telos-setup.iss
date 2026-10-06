; Inno Setup Script for Telos Game Client
; https://github.com/larvance/telos

#ifndef MyAppVersion
#define MyAppVersion "0.1.0"
#endif

#define MyAppName "Telos"
#define MyAppPublisher "Larvance"
#define MyAppURL "https://github.com/larvance/telos"
#define MyAppExeName "telos.exe"

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
OutputBaseFilename=telos-{#MyAppVersion}-windows-x86_64-setup
SetupIconFile=telos.ico
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
Source: "telos.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\assets\telos\*"; DestDir: "{app}\assets\telos"; Flags: ignoreversion recursesubdirs createallsubdirs; Flags: skipifsourcedoesntexist

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\telos.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\telos.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
