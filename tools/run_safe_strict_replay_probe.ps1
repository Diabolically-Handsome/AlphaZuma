[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [Parameter(Mandatory = $true)]
    [string]$Dmo,

    [Parameter(Mandatory = $true)]
    [string]$AppliedPreSnapshot,

    [int]$AttachAtUpdate = 0,
    [int]$StopAfterUpdate = 600,
    [uint32]$StartupCrtSeed = 23775218,
    [uint32]$BoardSeed = 1162643045,
    [int]$BoardSeedSkipCount = 0,
    [uint32]$GlobalRngSeed = 23775218,
    [uint32]$ThreadCrtRngSeed = 23775218,
    [int]$MaximumHits = 10000,
    [int]$TraceTimeoutSeconds = 180,
    [double]$ServiceWaitTimeoutSeconds = 60.0,
    [int]$StartupPriorityBiasUntilUpdate = -1,
    [ValidateSet('auto', 'debugger_register', 'iat_stub')]
    [string]$StartupSeedTransport = 'auto',
    [switch]$AllowAttachStabilization,
    [switch]$AllowPreAttachFileWriteDebt,
    [switch]$AllowFontCacheManifestCompletionDebt,
    [switch]$DisableBoardSeedControls,
    [string]$ProjectRoot = '',
    [string]$Python =
        'D:\ZumaGolden\tools\dxcam-venv\Scripts\python.exe',
    [string]$Runtime =
        'D:\ZumaGolden\tools\direct-runtime\popcapgame1.exe',
    [string]$Changedir =
        "D:\SteamLibrary\steamapps\common\Zuma's Revenge"
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (
        Split-Path -Parent $MyInvocation.MyCommand.Path
    )
}

function Assert-ExistingFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $LiteralPath -PathType Leaf)) {
        throw "$Label does not exist: $LiteralPath"
    }
}

function Get-ExactRuntimeProcesses {
    Get-CimInstance Win32_Process -Filter "Name='popcapgame1.exe'" |
        Where-Object { $_.ExecutablePath -eq $Runtime }
}

Assert-ExistingFile -LiteralPath $Dmo -Label 'DMO'
Assert-ExistingFile `
    -LiteralPath $AppliedPreSnapshot `
    -Label 'applied prestate snapshot'
Assert-ExistingFile -LiteralPath $Python -Label 'Python'
Assert-ExistingFile -LiteralPath $Runtime -Label 'direct runtime'
if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "project root does not exist: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $Changedir -PathType Container)) {
    throw "retail asset directory does not exist: $Changedir"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "output root already exists: $OutputRoot"
}
if (@(Get-ExactRuntimeProcesses).Count -ne 0) {
    throw 'the exact direct runtime is already running'
}
if (
    $AttachAtUpdate -lt 0 -or
    $StopAfterUpdate -lt 0 -or
    $AttachAtUpdate -ge $StopAfterUpdate
) {
    throw 'attach/stop updates are invalid'
}
if ($TraceTimeoutSeconds -le 0 -or $ServiceWaitTimeoutSeconds -le 0) {
    throw 'trace and service timeouts must be positive'
}
if ($BoardSeedSkipCount -lt 0) {
    throw 'board seed skip count must not be negative'
}
if (
    $StartupPriorityBiasUntilUpdate -lt -1 -or
    $StartupPriorityBiasUntilUpdate -eq 0
) {
    throw (
        'startup priority bias must be -1 (disabled) or a positive update'
    )
}
if (
    $AttachAtUpdate -eq 0 -and
    (
        $AllowAttachStabilization -or
        $AllowPreAttachFileWriteDebt -or
        $AllowFontCacheManifestCompletionDebt
    )
) {
    throw 'late-attach debt controls require a nonzero attach update'
}
if (
    $AllowFontCacheManifestCompletionDebt -and
    -not $AllowPreAttachFileWriteDebt
) {
    throw 'font-cache completion debt requires pre-attach file-write debt'
}

$stateTool = Join-Path $ProjectRoot 'tools\pc_state_transaction.py'
$traceTool = Join-Path $ProjectRoot 'tools\trace_popcap_demo_commands.py'
Assert-ExistingFile -LiteralPath $stateTool -Label 'state transaction tool'
Assert-ExistingFile -LiteralPath $traceTool -Label 'strict replay tool'

$nonce = [guid]::NewGuid().ToString('N')
$env:PYTHONPATH = "$(Join-Path $ProjectRoot 'src');$ProjectRoot"
$runError = $null
$cleanupError = $null
$traceExitCode = $null

New-Item -ItemType Directory -Path $OutputRoot | Out-Null

try {
    $hostPrePath = Join-Path $OutputRoot 'host-pre.json'
    & $Python $stateTool snapshot `
        --session-nonce $nonce `
        --phase host-pre `
        --output $hostPrePath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "host snapshot failed with exit code $LASTEXITCODE"
    }

    & $Python $stateTool restore --snapshot $AppliedPreSnapshot | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "prestate restore failed with exit code $LASTEXITCODE"
    }

    $appliedPath = Join-Path $OutputRoot 'applied-pre.json'
    & $Python $stateTool snapshot `
        --session-nonce $nonce `
        --phase applied-pre `
        --output $appliedPath | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "applied-pre snapshot failed with exit code $LASTEXITCODE"
    }

    $expectedPreRoot = (
        Get-Content -Raw -LiteralPath $AppliedPreSnapshot |
            ConvertFrom-Json
    ).state_root
    $appliedRoot = (
        Get-Content -Raw -LiteralPath $appliedPath |
            ConvertFrom-Json
    ).state_root
    if ($expectedPreRoot -ne $appliedRoot) {
        throw "applied prestate mismatch: $appliedRoot"
    }

    $startupSeedTransport = if ($StartupSeedTransport -ne 'auto') {
        $StartupSeedTransport
    } elseif ($AttachAtUpdate -eq 0) {
        'debugger_register'
    } else {
        'iat_stub'
    }
    $traceArguments = @(
        $traceTool,
        '--dmo', $Dmo,
        '--runtime-exe', $Runtime,
        '--direct-runtime-exe', $Runtime,
        '--changedir', $Changedir,
        '--crt-rand-seed', "$StartupCrtSeed",
        '--startup-seed-transport', $startupSeedTransport,
        '--maximum-hits', "$MaximumHits",
        '--trace-timeout', "$TraceTimeoutSeconds",
        '--attach-at-update', "$AttachAtUpdate",
        '--broker-service-blocks',
        '--service-wait-timeout', "$ServiceWaitTimeoutSeconds",
        '--quiet-nonservice',
        '--stop-after-update', "$StopAfterUpdate",
        '--progress-every-updates', '100',
        '--result-json', (Join-Path $OutputRoot 'result.json')
    )
    if (-not $DisableBoardSeedControls) {
        $traceArguments += @(
            '--board-seed-address', '6666280',
            '--board-seed', "$BoardSeed",
            '--board-seed-skip-count', "$BoardSeedSkipCount",
            '--global-rng-seed', "$GlobalRngSeed",
            '--thread-crt-rng-seed', "$ThreadCrtRngSeed"
        )
    }
    if ($AttachAtUpdate -eq 0) {
        $traceArguments += '--allow-pre-stream-commands'
    }
    if ($StartupPriorityBiasUntilUpdate -gt 0) {
        $traceArguments += @(
            '--startup-priority-bias-until-update',
            "$StartupPriorityBiasUntilUpdate"
        )
    }
    if ($AllowAttachStabilization) {
        $traceArguments += '--allow-attach-stabilization'
    }
    if ($AllowPreAttachFileWriteDebt) {
        $traceArguments += '--allow-pre-attach-file-write-debt'
    }
    if ($AllowFontCacheManifestCompletionDebt) {
        $traceArguments += '--allow-font-cache-manifest-completion-debt'
    }
    & $Python @traceArguments
    $traceExitCode = $LASTEXITCODE
    if ($traceExitCode -ne 0) {
        throw "strict replay failed with exit code $traceExitCode"
    }
}
catch {
    $runError = $_.Exception
}
finally {
    try {
        foreach ($process in @(Get-ExactRuntimeProcesses)) {
            Stop-Process `
                -Id ([int]$process.ProcessId) `
                -Force `
                -ErrorAction SilentlyContinue
        }
        Start-Sleep -Milliseconds 300

        $hostPrePath = Join-Path $OutputRoot 'host-pre.json'
        if (Test-Path -LiteralPath $hostPrePath -PathType Leaf) {
            & $Python $stateTool restore --snapshot $hostPrePath | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw "host restore failed with exit code $LASTEXITCODE"
            }

            $hostRestoredPath = Join-Path $OutputRoot 'host-restored.json'
            & $Python $stateTool snapshot `
                --session-nonce $nonce `
                --phase host-restored `
                --output $hostRestoredPath | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw (
                    "host-restored snapshot failed with exit code " +
                    "$LASTEXITCODE"
                )
            }

            $hostRoot = (
                Get-Content -Raw -LiteralPath $hostPrePath |
                    ConvertFrom-Json
            ).state_root
            $restoredRoot = (
                Get-Content -Raw -LiteralPath $hostRestoredPath |
                    ConvertFrom-Json
            ).state_root
            if ($hostRoot -ne $restoredRoot) {
                throw "host restore mismatch: $restoredRoot"
            }
        }
    }
    catch {
        $cleanupError = $_.Exception
    }
}

$hostPrePath = Join-Path $OutputRoot 'host-pre.json'
$hostRestoredPath = Join-Path $OutputRoot 'host-restored.json'
$summary = [ordered]@{
    output = $OutputRoot
    trace_result = Join-Path $OutputRoot 'result.json'
    trace_exit_code = $traceExitCode
    host_root = if (Test-Path -LiteralPath $hostPrePath) {
        (
            Get-Content -Raw -LiteralPath $hostPrePath |
                ConvertFrom-Json
        ).state_root
    } else {
        $null
    }
    restored_root = if (Test-Path -LiteralPath $hostRestoredPath) {
        (
            Get-Content -Raw -LiteralPath $hostRestoredPath |
                ConvertFrom-Json
        ).state_root
    } else {
        $null
    }
    run_error = if ($null -ne $runError) {
        $runError.Message
    } else {
        $null
    }
    cleanup_error = if ($null -ne $cleanupError) {
        $cleanupError.Message
    } else {
        $null
    }
}
$summary | ConvertTo-Json -Compress

if ($null -ne $cleanupError) {
    throw $cleanupError
}
if ($null -ne $runError) {
    throw $runError
}
