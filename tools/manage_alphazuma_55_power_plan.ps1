param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('InstallGuard', 'Guard', 'RequestRestore', 'Status')]
    [string]$Mode
)

$ErrorActionPreference = 'Stop'

$stateRoot = 'D:\ZumaTraining\alphazuma-55-weekend-s81081401-v1\hardware'
$heartbeatPath = Join-Path $stateRoot 'windows-power-plan-guard.json'
$restoreRequestPath = Join-Path $stateRoot 'power-plan-restore-requested.txt'
$trainingGuid = '8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c'
$restoreGuid = '381b4222-f694-41f0-9685-ff5bb260df2e'
$deadlineUtc = [datetime]::Parse(
    '2026-08-17T15:00:00Z',
    [Globalization.CultureInfo]::InvariantCulture,
    [Globalization.DateTimeStyles]::AdjustToUniversal
)

function Get-ActivePlan {
    $line = (& powercfg.exe /getactivescheme | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "powercfg /getactivescheme failed with exit code $LASTEXITCODE"
    }
    $match = [regex]::Match($line, '[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
    if (-not $match.Success) {
        throw "Could not parse active power scheme: $line"
    }
    return $match.Value.ToLowerInvariant()
}

function Set-Plan {
    param([Parameter(Mandatory = $true)][string]$Guid)
    & powercfg.exe /setactive $Guid
    if ($LASTEXITCODE -ne 0) {
        throw "powercfg /setactive failed for $Guid with exit code $LASTEXITCODE"
    }
    if ((Get-ActivePlan) -ne $Guid) {
        throw "active power scheme did not become $Guid"
    }
}

function Write-Receipt {
    param(
        [Parameter(Mandatory = $true)][string]$Status,
        [string]$ErrorMessage = ''
    )
    New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
    $value = [ordered]@{
        schema = 'zuma-rl.alphazuma-55-windows-power-plan-guard'
        version = 1
        campaign_id = 'alphazuma-55-weekend-s81081401-v1'
        status = $Status
        updated_utc = [datetime]::UtcNow.ToString('o')
        deadline_utc = $deadlineUtc.ToString('o')
        process_id = $PID
        training_guid = $trainingGuid
        restore_guid = $restoreGuid
        active_guid = Get-ActivePlan
        error = $ErrorMessage
    }
    $temporary = "$heartbeatPath.tmp-$PID"
    $value | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -Force -LiteralPath $temporary -Destination $heartbeatPath
}

if ($Mode -eq 'Status') {
    if (Test-Path -LiteralPath $heartbeatPath) {
        Get-Content -Raw -LiteralPath $heartbeatPath
    } else {
        [ordered]@{ status = 'NOT_INSTALLED'; active_guid = Get-ActivePlan } |
            ConvertTo-Json -Depth 4
    }
    exit 0
}

if ($Mode -eq 'RequestRestore') {
    New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
    [datetime]::UtcNow.ToString('o') | Set-Content -LiteralPath $restoreRequestPath -Encoding ascii
    exit 0
}

if ($Mode -eq 'InstallGuard') {
    New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue -LiteralPath $restoreRequestPath
    Set-Plan -Guid $trainingGuid
    Write-Receipt -Status 'TRAINING_PLAN_SET'
    $arguments = @(
        '-NoLogo', '-NoProfile', '-ExecutionPolicy', 'Bypass',
        '-File', ('"' + $PSCommandPath + '"'), '-Mode', 'Guard'
    )
    Start-Process -FilePath 'powershell.exe' -ArgumentList $arguments -WindowStyle Hidden
    exit 0
}

try {
    while ([datetime]::UtcNow -lt $deadlineUtc) {
        if (Test-Path -LiteralPath $restoreRequestPath) {
            Set-Plan -Guid $restoreGuid
            Write-Receipt -Status 'RESTORED_ON_REQUEST'
            exit 0
        }
        if ((Get-ActivePlan) -ne $trainingGuid) {
            Set-Plan -Guid $trainingGuid
        }
        Write-Receipt -Status 'GUARDING_TRAINING_PLAN'
        Start-Sleep -Seconds 60
    }
    Set-Plan -Guid $restoreGuid
    Write-Receipt -Status 'RESTORED_AT_DEADLINE'
} catch {
    try {
        Write-Receipt -Status 'ERROR' -ErrorMessage $_.Exception.Message
    } catch {
        # Preserve the original exception if receipt generation also fails.
    }
    throw
}
