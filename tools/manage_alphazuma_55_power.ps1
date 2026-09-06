param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('InstallTrainingGuard', 'Guard', 'RequestRestore', 'Status')]
    [string]$Mode
)

$ErrorActionPreference = 'Stop'

$campaignId = 'alphazuma-55-weekend-s81081401-v1'
$stateRoot = 'D:\ZumaTraining\alphazuma-55-weekend-s81081401-v1\hardware'
$heartbeatPath = Join-Path $stateRoot 'gpu-power-guard.json'
$stopPath = Join-Path $stateRoot 'restore-requested.txt'
$deadlineUtc = [datetime]::Parse(
    '2026-08-17T15:00:00Z',
    [Globalization.CultureInfo]::InvariantCulture,
    [Globalization.DateTimeStyles]::AdjustToUniversal
)
$trainingLimits = [ordered]@{ '0' = 550; '1' = 250 }
$restoreLimits = [ordered]@{ '0' = 600; '1' = 400 }

function Get-NvidiaSmi {
    $command = Get-Command nvidia-smi.exe -ErrorAction Stop
    return $command.Source
}

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Get-GpuState {
    param([Parameter(Mandatory = $true)][string]$NvidiaSmi)

    $lines = & $NvidiaSmi `
        --query-gpu=index,name,power.limit,power.min_limit,power.max_limit,temperature.gpu,utilization.gpu `
        --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi query failed with exit code $LASTEXITCODE"
    }
    $rows = foreach ($line in $lines) {
        $parts = $line -split '\s*,\s*'
        [ordered]@{
            index = [int]$parts[0]
            name = $parts[1]
            power_limit_watts = [double]$parts[2]
            min_limit_watts = [double]$parts[3]
            max_limit_watts = [double]$parts[4]
            temperature_c = [int]$parts[5]
            utilization_percent = [int]$parts[6]
        }
    }
    return @($rows)
}

function Set-GpuLimits {
    param(
        [Parameter(Mandatory = $true)][string]$NvidiaSmi,
        [Parameter(Mandatory = $true)][System.Collections.IDictionary]$Limits
    )

    foreach ($entry in $Limits.GetEnumerator()) {
        & $NvidiaSmi -i ([string]$entry.Key) -pl ([string]$entry.Value)
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to set GPU $($entry.Key) to $($entry.Value) W"
        }
    }
}

function Write-GuardReceipt {
    param(
        [Parameter(Mandatory = $true)][string]$Status,
        [Parameter(Mandatory = $true)][string]$NvidiaSmi,
        [string]$ErrorMessage = ''
    )

    New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
    $payload = [ordered]@{
        schema = 'zuma-rl.alphazuma-55-gpu-power-guard'
        version = 1
        campaign_id = $campaignId
        status = $Status
        updated_utc = [datetime]::UtcNow.ToString('o')
        deadline_utc = $deadlineUtc.ToString('o')
        process_id = $PID
        elevated = Test-IsAdministrator
        training_limits_watts = $trainingLimits
        restore_limits_watts = $restoreLimits
        gpu_state = @(Get-GpuState -NvidiaSmi $NvidiaSmi)
        error = $ErrorMessage
    }
    $temporaryPath = "$heartbeatPath.tmp-$PID"
    $payload | ConvertTo-Json -Depth 8 | Set-Content `
        -LiteralPath $temporaryPath -Encoding utf8
    Move-Item -Force -LiteralPath $temporaryPath -Destination $heartbeatPath
}

$nvidiaSmi = Get-NvidiaSmi

if ($Mode -eq 'Status') {
    if (Test-Path -LiteralPath $heartbeatPath) {
        Get-Content -Raw -LiteralPath $heartbeatPath
    } else {
        [ordered]@{
            status = 'NOT_INSTALLED'
            gpu_state = @(Get-GpuState -NvidiaSmi $nvidiaSmi)
        } | ConvertTo-Json -Depth 6
    }
    exit 0
}

if ($Mode -eq 'RequestRestore') {
    New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
    [datetime]::UtcNow.ToString('o') | Set-Content `
        -LiteralPath $stopPath -Encoding ascii
    exit 0
}

if (-not (Test-IsAdministrator)) {
    throw "$Mode requires an elevated Windows administrator token"
}

if ($Mode -eq 'InstallTrainingGuard') {
    New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue -LiteralPath $stopPath
    Set-GpuLimits -NvidiaSmi $nvidiaSmi -Limits $trainingLimits
    Write-GuardReceipt -Status 'TRAINING_LIMITS_SET' -NvidiaSmi $nvidiaSmi

    $arguments = @(
        '-NoLogo',
        '-NoProfile',
        '-ExecutionPolicy', 'Bypass',
        '-File', ('"' + $PSCommandPath + '"'),
        '-Mode', 'Guard'
    )
    Start-Process -FilePath 'powershell.exe' `
        -ArgumentList $arguments -WindowStyle Hidden
    exit 0
}

try {
    Set-GpuLimits -NvidiaSmi $nvidiaSmi -Limits $trainingLimits
    while ([datetime]::UtcNow -lt $deadlineUtc) {
        if (Test-Path -LiteralPath $stopPath) {
            Set-GpuLimits -NvidiaSmi $nvidiaSmi -Limits $restoreLimits
            Write-GuardReceipt -Status 'RESTORED_ON_REQUEST' -NvidiaSmi $nvidiaSmi
            exit 0
        }

        $state = @(Get-GpuState -NvidiaSmi $nvidiaSmi)
        $needsRepair = $false
        foreach ($entry in $trainingLimits.GetEnumerator()) {
            $row = $state | Where-Object { $_.index -eq [int]$entry.Key }
            if ($null -eq $row -or [double]$row.power_limit_watts -ne [double]$entry.Value) {
                $needsRepair = $true
            }
        }
        if ($needsRepair) {
            Set-GpuLimits -NvidiaSmi $nvidiaSmi -Limits $trainingLimits
        }
        Write-GuardReceipt -Status 'GUARDING_TRAINING_LIMITS' -NvidiaSmi $nvidiaSmi
        Start-Sleep -Seconds 60
    }

    Set-GpuLimits -NvidiaSmi $nvidiaSmi -Limits $restoreLimits
    Write-GuardReceipt -Status 'RESTORED_AT_DEADLINE' -NvidiaSmi $nvidiaSmi
} catch {
    try {
        Write-GuardReceipt `
            -Status 'ERROR' `
            -NvidiaSmi $nvidiaSmi `
            -ErrorMessage $_.Exception.Message
    } catch {
        # Preserve the original failure if even receipt generation is unavailable.
    }
    throw
}
