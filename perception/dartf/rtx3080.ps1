param(
    [ValidateSet('Image', 'Login', 'Check', 'Download', 'Build', 'Test', 'Live')]
    [string]$Action = 'Check',
    [string]$Video,
    [string]$Prompt = 'pen',
    [ValidateRange(1, 1000000)][int]$Frames = 300,
    [ValidateRange(1, 128)][int]$Threads = 8,
    [string]$WorkDir,
    [switch]$Render,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$perceptionRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $WorkDir) { $WorkDir = Join-Path $perceptionRoot 'dartf-local\rtx3080' }
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
$cacheDir = Join-Path $WorkDir 'hf'
$assetsDir = Join-Path $WorkDir 'assets'
$imageName = 'relish-dartf:sm86'

function Invoke-Docker([string[]]$Arguments) {
    if ($DryRun) {
        # JSON preserves argument boundaries for the runnable launcher check.
        ConvertTo-Json -InputObject $Arguments -Compress
        return
    }
    & docker @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed (exit $LASTEXITCODE). See output above." }
}

if ($Action -eq 'Image') {
    Invoke-Docker @('build', '-t', 'relish-dartf:local', $PSScriptRoot)
    Invoke-Docker @('build', '-t', $imageName, '-f', (Join-Path $PSScriptRoot 'Dockerfile.fast'), $PSScriptRoot)
    exit
}

if ($Action -eq 'Test' -or $Action -eq 'Live') {
    if ([string]::IsNullOrWhiteSpace($Prompt) -or $Prompt.Contains(',')) {
        throw 'Use a single non-empty prompt, such as pen.'
    }
}
if ($Action -eq 'Test') {
    if (-not $Video -or -not (Test-Path -LiteralPath $Video -PathType Leaf)) {
        throw 'Test requires -Video pointing to a recorded video file.'
    }
    $Video = (Resolve-Path -LiteralPath $Video).Path
}
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $cacheDir, $assetsDir | Out-Null
}

$dockerArgs = @('run', '--rm', '--name', ('relish-dartf-fast-' + [Guid]::NewGuid().ToString('N').Substring(0, 12)))
if ($Action -eq 'Login') {
    Invoke-Docker ($dockerArgs + @('-it', '--mount', "type=bind,source=$cacheDir,target=/hf",
        '--entrypoint', 'hf', $imageName, 'auth', 'login'))
    exit
}
if ($Action -ne 'Download') { $dockerArgs += @('--gpus', 'all') }
if ($Action -eq 'Live') { $dockerArgs += '-i' }
$dockerArgs += @('--mount', "type=bind,source=$cacheDir,target=/hf",
    '--mount', "type=bind,source=$assetsDir,target=/assets",
    '--mount', "type=bind,source=$perceptionRoot,target=/app,readonly")
if ($Action -eq 'Test') {
    $inputDir = Split-Path -Parent $Video
    $inputName = Split-Path -Leaf $Video
    $dockerArgs += @('--mount', "type=bind,source=$inputDir,target=/input,readonly")
}
$stages = @{ Check = 'check'; Download = 'download'; Build = 'all'; Test = 'benchmark' }
if ($Action -eq 'Live') {
    $dockerArgs += @('--entrypoint', 'python', $imageName, '/app/dartf/live.py')
} else {
    $dockerArgs += @('--entrypoint', 'python', $imageName, '/app/dartf/fast.py',
        '--stage', $stages[$Action], '--threads', "$Threads")
}
if ($Action -eq 'Test') {
    $dockerArgs += @('--video', "/input/$inputName", '--prompt', $Prompt, '--frames', "$Frames")
    if ($Render) { $dockerArgs += '--render' }
}
if ($Action -eq 'Live') {
    $dockerArgs += @('--prompt', $Prompt, '--frames', "$Frames")
}
Invoke-Docker $dockerArgs
if (-not $DryRun) { Write-Host "Local files: $WorkDir" }
