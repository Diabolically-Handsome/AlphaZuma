param(
    [Parameter(Mandatory = $true)]
    [DateTimeOffset]$DeadlineUtc,

    [Parameter(Mandatory = $true)]
    [string]$ReceiptPath
)

$ErrorActionPreference = 'Stop'
$balancedGuid = '381b4222-f694-41f0-9685-ff5bb260df2e'
$expectedLimits = @(600.0, 400.0)

function Get-GpuSnapshot {
    $rows = & nvidia-smi.exe --query-gpu=index,name,power.limit,power.draw,temperature.gpu --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi query failed with exit code $LASTEXITCODE"
    }
    $parsed = @()
    foreach ($row in $rows) {
        $parts = $row -split ',' | ForEach-Object { $_.Trim() }
        if ($parts.Count -ne 5) {
            throw "unexpected nvidia-smi row: $row"
        }
        $parsed += [ordered]@{
            index = [int]$parts[0]
            name = $parts[1]
            power_limit_watts = [double]$parts[2]
            power_draw_watts = [double]$parts[3]
            temperature_c = [int]$parts[4]
        }
    }
    return $parsed
}

function Get-ActiveScheme {
    $value = (& powercfg.exe /getactivescheme | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "powercfg /getactivescheme failed with exit code $LASTEXITCODE"
    }
    return $value
}

while ([DateTimeOffset]::UtcNow -lt $DeadlineUtc.ToUniversalTime()) {
    $remaining = ($DeadlineUtc.ToUniversalTime() - [DateTimeOffset]::UtcNow).TotalSeconds
    $delay = [Math]::Max(1, [Math]::Min(30, [Math]::Floor($remaining)))
    Start-Sleep -Seconds $delay
}

$observedUtc = [DateTimeOffset]::UtcNow.ToString('o')
$beforeGpu = Get-GpuSnapshot
$beforeScheme = Get-ActiveScheme
$commands = @()
$errors = @()

foreach ($row in @(
    @{ index = 0; watts = 600 },
    @{ index = 1; watts = 400 }
)) {
    $output = (& nvidia-smi.exe -i $row.index -pl $row.watts 2>&1 | Out-String).Trim()
    $code = $LASTEXITCODE
    $commands += [ordered]@{
        command = "nvidia-smi.exe -i $($row.index) -pl $($row.watts)"
        exit_code = $code
        output = $output
    }
    if ($code -ne 0) {
        $errors += "GPU $($row.index) power restore failed with exit code $code"
    }
}

$planOutput = (& powercfg.exe /setactive $balancedGuid 2>&1 | Out-String).Trim()
$planCode = $LASTEXITCODE
$commands += [ordered]@{
    command = "powercfg.exe /setactive $balancedGuid"
    exit_code = $planCode
    output = $planOutput
}
if ($planCode -ne 0) {
    $errors += "balanced power plan restore failed with exit code $planCode"
}

$afterGpu = Get-GpuSnapshot
$afterScheme = Get-ActiveScheme
$limits = @($afterGpu | Sort-Object index | ForEach-Object { [double]$_.power_limit_watts })
if ($limits.Count -ne 2 -or $limits[0] -ne $expectedLimits[0] -or $limits[1] -ne $expectedLimits[1]) {
    $errors += "verified GPU power limits differ: $($limits -join ',')"
}
if ($afterScheme.ToLowerInvariant() -notlike "*$balancedGuid*") {
    $errors += "verified active power scheme is not Balanced: $afterScheme"
}

$scriptHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $PSCommandPath).Hash.ToLowerInvariant()
$receipt = [ordered]@{
    schema = 'zuma-rl.alphazuma-55-weekend-deadline-power-restore-receipt'
    version = 1
    status = if ($errors.Count -eq 0) { 'PASS' } else { 'ERROR' }
    deadline_utc = $DeadlineUtc.ToUniversalTime().ToString('o')
    observed_utc = $observedUtc
    completed_utc = [DateTimeOffset]::UtcNow.ToString('o')
    watcher = [ordered]@{
        path = $PSCommandPath
        sha256 = "sha256:$scriptHash"
        pid = $PID
    }
    before = [ordered]@{
        gpus = $beforeGpu
        active_power_scheme = $beforeScheme
    }
    commands = $commands
    after = [ordered]@{
        gpus = $afterGpu
        active_power_scheme = $afterScheme
    }
    verification = [ordered]@{
        expected_gpu_power_limits_watts = $expectedLimits
        expected_balanced_power_scheme_guid = $balancedGuid
        passed = ($errors.Count -eq 0)
        errors = $errors
    }
    training_or_evaluation_process_signaled = $false
    model_or_recipe_change = $false
    historical_data_touched = $false
}

$resolvedReceipt = [System.IO.Path]::GetFullPath($ReceiptPath)
$parent = [System.IO.Path]::GetDirectoryName($resolvedReceipt)
[System.IO.Directory]::CreateDirectory($parent) | Out-Null
$json = $receipt | ConvertTo-Json -Depth 12
$bytes = [System.Text.UTF8Encoding]::new($false).GetBytes("$json`n")
$stream = [System.IO.File]::Open(
    $resolvedReceipt,
    [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write,
    [System.IO.FileShare]::Read
)
try {
    $stream.Write($bytes, 0, $bytes.Length)
} finally {
    $stream.Dispose()
}

if ($errors.Count -ne 0) {
    throw ($errors -join '; ')
}
