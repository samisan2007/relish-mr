param(
    [ValidateSet('Setup', 'Track', 'Image')][string]$Action = 'Track',
    [ValidateSet('edgetam', 'sam21tiny', 'edgetam-hf', 'evm')][string]$Model = 'edgetam',
    [string]$InputPath,
    [string]$Prompt = 'meatball',
    [ValidateRange(1, 16)][int]$Objects = 3,
    [ValidateRange(1, 10)][int]$Repeats = 2,
    [ValidateRange(0.0, 1.0)][double]$Threshold = 0.4
)

$ErrorActionPreference = 'Stop'
$perceptionRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$localRoot = Join-Path $perceptionRoot 'candidates-local'
$imageName = 'relish-candidates:local'
$models = Get-Content -Raw (Join-Path $PSScriptRoot 'models.json') | ConvertFrom-Json

if ($Action -eq 'Setup') {
    New-Item -ItemType Directory -Force $localRoot | Out-Null
    foreach ($entry in $models.PSObject.Properties) {
        $spec = $entry.Value
        if (-not $spec.source) { continue }
        $sourceRoot = Join-Path $localRoot $spec.directory
        if (-not (Test-Path -LiteralPath $sourceRoot)) {
            & git clone $spec.source $sourceRoot
            if ($LASTEXITCODE -ne 0) { throw "Clone failed: $($entry.Name)" }
            & git -c "safe.directory=$sourceRoot" -C $sourceRoot checkout $spec.revision
            if ($LASTEXITCODE -ne 0) { throw "Checkout failed: $($entry.Name)" }
        }
        $revision = & git -c "safe.directory=$sourceRoot" -C $sourceRoot rev-parse HEAD
        if ($LASTEXITCODE -ne 0 -or $revision -ne $spec.revision) {
            throw "Unexpected source revision for $($entry.Name); preserve it and use a fresh checkout."
        }
    }
    & docker build -t $imageName $PSScriptRoot
    if ($LASTEXITCODE -ne 0) { throw 'Image build failed. Prepare relish-sam31:local first (sam31/run.ps1 -Action Setup).' }
} elseif (-not $InputPath -or -not (Test-Path -LiteralPath $InputPath)) {
    throw 'Provide -InputPath: a prepared case directory for Track, or a photo for Image.'
}

$imageId = & docker image inspect --format '{{.Id}}' $imageName
if ($LASTEXITCODE -ne 0) { throw 'Candidate image missing; run -Action Setup first.' }
$dockerArgs = @('run', '--rm', '--gpus', 'all', '--env', "CANDIDATE_IMAGE_ID=$imageId",
    '--name', ('relish-candidates-' + [Guid]::NewGuid().ToString('N').Substring(0, 12)),
    '--mount', "type=bind,source=$perceptionRoot,target=/app,readonly",
    '--mount', "type=bind,source=$localRoot,target=/local",
    '--entrypoint', 'python', '--env', 'OMP_NUM_THREADS=8')
if ($Action -eq 'Setup') {
    $dockerArgs += @($imageName, '/app/candidates/benchmark.py', 'setup')
} else {
    $inputResolved = (Resolve-Path -LiteralPath $InputPath).Path
    if ($Action -eq 'Track') {
        if ($Model -eq 'evm') { throw 'EV-M is an image detector; use -Action Image.' }
        if (-not (Test-Path -LiteralPath $inputResolved -PathType Container)) { throw 'Track requires a prepared case directory.' }
        $dockerArgs += @('--mount', "type=bind,source=$inputResolved,target=/input,readonly")
        $workerArgs = @('track', '--model', $Model, '--input', '/input', '--objects', "$Objects")
    } else {
        if ($Model -ne 'evm') { throw 'Image tests use -Model evm; trackers require prepared seed masks.' }
        if (-not (Test-Path -LiteralPath $inputResolved -PathType Leaf)) { throw 'Image requires a photo.' }
        $dockerArgs += @('--mount', "type=bind,source=$(Split-Path -Parent $inputResolved),target=/input,readonly")
        $workerArgs = @('image', '--model', $Model, '--input', "/input/$(Split-Path -Leaf $inputResolved)",
                        '--prompt', $Prompt, '--threshold', $Threshold.ToString([Globalization.CultureInfo]::InvariantCulture))
    }
    $dockerArgs += @('--network', 'none', '--env', 'HF_HUB_OFFLINE=1', $imageName,
                      '/app/candidates/benchmark.py') + $workerArgs + @('--repeats', "$Repeats")
}
& docker @dockerArgs
if ($LASTEXITCODE -ne 0) { throw "Candidate $Action failed (exit $LASTEXITCODE)." }
Write-Host "Local assets and results: $localRoot"
