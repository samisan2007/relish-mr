param(
    [ValidateSet('Setup', 'Test')][string]$Action = 'Test',
    [string]$InputPath,
    [string]$Prompt = 'meatball',
    [ValidateRange(10, 120)][int]$Frames = 60,
    [ValidateRange(1, 16)][int]$MaxObjects = 16,
    [ValidateRange(1, 10)][int]$Repeats = 2,
    [switch]$Compile,
    [switch]$Render
)

$ErrorActionPreference = 'Stop'
$perceptionRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$localRoot = Join-Path $perceptionRoot 'sam31-local'
$sourceRoot = Join-Path $localRoot 'source'
$sourceRevision = '660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7'
$checkpointRevision = 'daa63191845a41281374e725f4c9e51c7a824460'
$imageName = 'relish-sam31:local'

if ($Action -eq 'Setup') {
    New-Item -ItemType Directory -Force -Path $localRoot | Out-Null
    $baseImage = & docker image ls --quiet --filter 'reference=relish-dartf:local'
    if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop must be running Linux containers.' }
    if (-not $baseImage) {
        & docker build -t 'relish-dartf:local' (Join-Path $perceptionRoot 'dartf')
        if ($LASTEXITCODE -ne 0) { throw 'Could not build the GPU environment.' }
    }
    & docker build -t $imageName $PSScriptRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not build the SAM 3.1 test environment.' }
    if (-not (Test-Path -LiteralPath $sourceRoot)) {
        & git clone https://github.com/facebookresearch/sam3.git $sourceRoot
        if ($LASTEXITCODE -ne 0) { throw 'Could not download SAM 3.1 source.' }
        & git -C $sourceRoot checkout $sourceRevision
        if ($LASTEXITCODE -ne 0) { throw 'Could not select the pinned SAM 3.1 source.' }
    }
}
if (-not (Test-Path -LiteralPath $sourceRoot)) { throw 'Run -Action Setup first.' }

$dockerArgs = @('run', '--rm', '--name', ('relish-sam31-' + [Guid]::NewGuid().ToString('N').Substring(0, 12)),
    '--mount', "type=bind,source=$localRoot,target=/local",
    '--mount', "type=bind,source=$sourceRoot,target=/sam31,readonly",
    '--mount', "type=bind,source=$perceptionRoot,target=/app,readonly",
    '--env', 'HF_HOME=/local/hf', '--env', 'PYTHONPATH=/sam31',
    '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'OMP_NUM_THREADS=8',
    '--env', 'TORCHINDUCTOR_CACHE_DIR=/local/inductor', '--env', 'TRITON_CACHE_DIR=/local/triton')
if ($Action -eq 'Setup') {
    $tokenPath = Join-Path $env:USERPROFILE '.cache\huggingface\token'
    if (Test-Path -LiteralPath $tokenPath) {
        $dockerArgs += @('--mount', "type=bind,source=$tokenPath,target=/hf-token,readonly", '--env', 'HF_TOKEN_PATH=/hf-token')
    }
    $dockerArgs += @('--entrypoint', 'hf', $imageName, 'download', 'facebook/sam3.1',
        'sam3.1_multiplex.pt', '--revision', $checkpointRevision)
} else {
    if (-not $InputPath -or -not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
        throw 'Test requires -InputPath pointing to an image or recorded video.'
    }
    if ([string]::IsNullOrWhiteSpace($Prompt)) { throw 'Provide a prompt.' }
    $inputFile = (Resolve-Path -LiteralPath $InputPath).Path
    $dockerArgs += @('--gpus', 'all', '--env', 'HF_HUB_OFFLINE=1',
        '--mount', "type=bind,source=$(Split-Path -Parent $inputFile),target=/input,readonly",
        '--entrypoint', 'python', $imageName, '/app/sam31/benchmark.py', "/input/$(Split-Path -Leaf $inputFile)",
        '--prompt', $Prompt, '--frames', "$Frames", '--max-objects', "$MaxObjects", '--repeats', "$Repeats")
    if ($Compile) { $dockerArgs += '--compile' }
    if ($Render) { $dockerArgs += '--render' }
}
& docker @dockerArgs
if ($LASTEXITCODE -ne 0) { throw "SAM 3.1 command failed (exit $LASTEXITCODE). See output above." }
Write-Host "Local files: $localRoot"
