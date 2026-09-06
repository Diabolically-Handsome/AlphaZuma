[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [Parameter(Mandatory = $true)]
    [string]$Dmo,

    [Parameter(Mandatory = $true)]
    [string]$ReferenceRngLog,

    [Parameter(Mandatory = $true)]
    [uint32]$GlobalMtrandSeed,

    [Parameter(Mandatory = $true)]
    [int]$GlobalSourceUpdate,

    [Parameter(Mandatory = $true)]
    [int]$GlobalRestoreRewindDraws,

    [Parameter(Mandatory = $true)]
    [int]$ThreadCrtSourceUpdate,

    [Parameter(Mandatory = $true)]
    [int]$ThreadCrtRestoreRewindDraws,

    [Parameter(Mandatory = $true)]
    [int]$RestoreCommandOrder,

    [int]$GlobalRestoreCommandOrder = -1,
    [int]$ThreadCrtRestoreCommandOrder = -1,

    [Parameter(Mandatory = $true)]
    [int]$StopAfterUpdate,

    [Parameter(Mandatory = $true)]
    [int]$StopAfterNativeGameTime,

    [string]$ProjectRoot = '',
    [string]$Python = 'D:\ZumaGolden\tools\dxcam-venv\Scripts\python.exe',
    [string]$Runtime = 'D:\ZumaGolden\tools\direct-runtime\popcapgame1.exe',
    [string]$Changedir = "D:\SteamLibrary\steamapps\common\Zuma's Revenge",
    [string]$RecordPreSnapshot =
        'D:\ZumaGolden\controlled-natural-win-r3-fullrng-20260731\safety\record-pre.json',
    [uint32]$StartupCrtSeed = 147023375,
    [ValidateSet('debugger_register', 'iat_stub')]
    [string]$StartupSeedTransport = 'iat_stub',
    [switch]$StartupTraceHandoff,
    [int]$AttachAtUpdate = 0,
    [int]$StartupPriorityBiasUntilUpdate = 500,
    [double]$ServiceWaitTimeoutSeconds = 15.0,
    [switch]$DisableBoundaryRngRestores,
    [switch]$AllowAttachStabilization,
    [switch]$AllowPreAttachFileWriteDebt,
    [switch]$AllowFontCacheManifestCompletionDebt,
    [int]$MaximumHits = 100000,
    [int]$TraceTimeoutSeconds = 600,
    [int]$MonitorTimeoutSeconds = 180,
    [double]$MonitorIntervalSeconds = 0.001,
    [int]$GlobalCallTraceStartUpdate = -1,
    [int]$GlobalCallTraceEndUpdate = -1,
    [int]$GlobalCallTraceTimeoutSeconds = 180,
    [switch]$HardwareGlobalCallTrace,
    [int]$HardwareGlobalCallTraceTimeoutSeconds = 300,
    [int]$HardwareGlobalCallTraceMaximumCalls = 100000,
    [int]$HardwareGlobalCallTraceLaunchAfterUpdate = -1,
    [int]$GlobalCallRestoreSourceUpdate = -1,
    [int]$GlobalCallRestoreFrameworkUpdate = -1,
    [uint32]$GlobalCallRestoreCaller = 0,
    [int]$GlobalCallRestoreRewindDraws = 0,
    [string]$GlobalCallRestoreTrace = '',
    [string]$GlobalCallRestoreRecordingReport = '',
    [int]$GlobalCallRestoreSourceOrder = -1,
    [string]$ThreadCrtCallRestoreTrace = '',
    [string]$ThreadCrtCallRestoreRecordingReport = '',
    [int]$ThreadCrtCallRestoreSourceOrder = -1,
    [int]$ThreadCrtCallRestoreFrameworkUpdate = -1,
    [uint32]$ThreadCrtCallRestoreCaller = 0,
    [int]$FrameMtrandSyncStartUpdate = -1,
    [int]$FrameMtrandSyncEndUpdate = -1,
    [int]$FrameMtrandSyncMaximumDraws = 20000,
    [int]$FrameMtrandSyncTimeoutSeconds = 300,
    [string]$GameplayMtrandOracle = '',
    [int]$GameplayMtrandMaximumDraws = 20000,
    [int]$GameplayMtrandSyncTimeoutSeconds = 600,
    [string]$GlobalMtrandCallOracle = '',
    [int]$GlobalMtrandCallStartAfterUpdate = -1,
    [int]$GlobalMtrandCallEndAtUpdate = -1,
    [int]$GlobalMtrandCallMaximumDraws = 20000,
    [int]$GlobalMtrandCallSyncTimeoutSeconds = 600,
    [switch]$WaitForNaturalRuntimeExit,
    [switch]$HeadlessFullReplay,
    [switch]$RequireSourceBoundGameplayParity,
    [int]$RuntimeExitTimeoutSeconds = 180,
    [int]$MinimumWinScore = 9650,
    [ValidateSet('natural_loss', 'natural_win')]
    [string]$ExpectedOutcome = 'natural_win'
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (
        Split-Path -Parent $MyInvocation.MyCommand.Path
    )
}
if ($GlobalRestoreCommandOrder -lt 0) {
    $GlobalRestoreCommandOrder = $RestoreCommandOrder
}
if ($ThreadCrtRestoreCommandOrder -lt 0) {
    $ThreadCrtRestoreCommandOrder = $RestoreCommandOrder
}
if (
    $AttachAtUpdate -lt 0 -or
    $StartupPriorityBiasUntilUpdate -lt -1 -or
    $StartupPriorityBiasUntilUpdate -eq 0 -or
    (
        $AttachAtUpdate -gt 0 -and
        $StartupPriorityBiasUntilUpdate -gt 0 -and
        $StartupPriorityBiasUntilUpdate -ge $AttachAtUpdate
    ) -or
    (
        $AttachAtUpdate -eq 0 -and
        (
            $AllowAttachStabilization -or
            $AllowPreAttachFileWriteDebt -or
            (
                $AllowFontCacheManifestCompletionDebt -and
                -not $StartupTraceHandoff
            )
        )
    ) -or
    (
        $AllowFontCacheManifestCompletionDebt -and
        -not (
            $AllowPreAttachFileWriteDebt -or $StartupTraceHandoff
        )
    ) -or
    $ServiceWaitTimeoutSeconds -le 0
) {
    throw 'late trace attachment options are invalid'
}
if (
    $StartupTraceHandoff -and
    (
        $AttachAtUpdate -ne 0 -or
        $StartupSeedTransport -ne 'debugger_register' -or
        $StartupPriorityBiasUntilUpdate -gt 0 -or
        $AllowAttachStabilization -or
        $AllowPreAttachFileWriteDebt -or
        -not $AllowFontCacheManifestCompletionDebt
    )
) {
    throw 'startup trace handoff options are invalid'
}
$globalCallTraceRequested = (
    $GlobalCallTraceStartUpdate -ge 0 -or
    $GlobalCallTraceEndUpdate -ge 0
)
$hardwareGlobalCallTraceRequested = [bool]$HardwareGlobalCallTrace
$postGlobalCallTraceRequested = (
    $globalCallTraceRequested -or $hardwareGlobalCallTraceRequested
)
if ($globalCallTraceRequested -and $hardwareGlobalCallTraceRequested) {
    throw 'software and hardware global call tracing are mutually exclusive'
}
if (
    $hardwareGlobalCallTraceRequested -and
    (
        $HardwareGlobalCallTraceTimeoutSeconds -le 0 -or
        $HardwareGlobalCallTraceMaximumCalls -le 0 -or
        (
            $HardwareGlobalCallTraceLaunchAfterUpdate -ge 0 -and
            (
                $HardwareGlobalCallTraceLaunchAfterUpdate -le
                    $AttachAtUpdate -or
                $HardwareGlobalCallTraceLaunchAfterUpdate -ge
                    $StopAfterUpdate
            )
        )
    )
) {
    throw 'hardware global call trace bounds are invalid'
}
if (
    $globalCallTraceRequested -and
    (
        $GlobalCallTraceStartUpdate -lt 0 -or
        $GlobalCallTraceEndUpdate -lt $GlobalCallTraceStartUpdate -or
        $StopAfterUpdate -ge $GlobalCallTraceStartUpdate
    )
) {
    throw (
        'global call trace requires a valid range strictly after the ' +
        'DMO trace stop update'
    )
}
$globalCallRestoreRequested = (
    $GlobalCallRestoreSourceUpdate -ge 0 -or
    $GlobalCallRestoreFrameworkUpdate -ge 0 -or
    $GlobalCallRestoreCaller -ne 0 -or
    -not [string]::IsNullOrWhiteSpace($GlobalCallRestoreTrace) -or
    -not [string]::IsNullOrWhiteSpace(
        $GlobalCallRestoreRecordingReport
    ) -or
    $GlobalCallRestoreSourceOrder -ge 0
)
if (
    $globalCallRestoreRequested -and
    (
        -not $postGlobalCallTraceRequested -or
        $GlobalCallRestoreSourceUpdate -lt 0 -or
        $GlobalCallRestoreCaller -eq 0 -or
        $GlobalCallRestoreRewindDraws -lt 0 -or
        (
            $globalCallTraceRequested -and
            (
                $GlobalCallRestoreFrameworkUpdate -lt
                    $GlobalCallTraceStartUpdate -or
                $GlobalCallRestoreFrameworkUpdate -gt
                    $GlobalCallTraceEndUpdate
            )
        ) -or
        (
            $hardwareGlobalCallTraceRequested -and
            (
                $GlobalCallRestoreFrameworkUpdate -le $StopAfterUpdate -or
                [string]::IsNullOrWhiteSpace(
                    $GlobalCallRestoreTrace
                ) -or
                [string]::IsNullOrWhiteSpace(
                    $GlobalCallRestoreRecordingReport
                ) -or
                $GlobalCallRestoreSourceOrder -lt 0
            )
        )
    )
) {
    throw 'global call-site restore options are incomplete or out of range'
}
$threadCrtCallRestoreRequested = (
    -not [string]::IsNullOrWhiteSpace($ThreadCrtCallRestoreTrace) -or
    -not [string]::IsNullOrWhiteSpace(
        $ThreadCrtCallRestoreRecordingReport
    ) -or
    $ThreadCrtCallRestoreSourceOrder -ge 0 -or
    $ThreadCrtCallRestoreFrameworkUpdate -ge 0 -or
    $ThreadCrtCallRestoreCaller -ne 0
)
if (
    $threadCrtCallRestoreRequested -and
    (
        -not $postGlobalCallTraceRequested -or
        [string]::IsNullOrWhiteSpace($ThreadCrtCallRestoreTrace) -or
        [string]::IsNullOrWhiteSpace(
            $ThreadCrtCallRestoreRecordingReport
        ) -or
        $ThreadCrtCallRestoreSourceOrder -lt 0 -or
        (
            $globalCallTraceRequested -and
            (
                $ThreadCrtCallRestoreFrameworkUpdate -lt
                    $GlobalCallTraceStartUpdate -or
                $ThreadCrtCallRestoreFrameworkUpdate -gt
                    $GlobalCallTraceEndUpdate
            )
        ) -or
        (
            $hardwareGlobalCallTraceRequested -and
            $ThreadCrtCallRestoreFrameworkUpdate -le $StopAfterUpdate
        ) -or
        $ThreadCrtCallRestoreCaller -eq 0
    )
) {
    throw 'thread CRT call-site restore options are incomplete or out of range'
}
$frameMtrandSyncRequested = (
    $FrameMtrandSyncStartUpdate -ge 0 -or
    $FrameMtrandSyncEndUpdate -ge 0
)
if (
    $frameMtrandSyncRequested -and
    (
        $FrameMtrandSyncStartUpdate -le 0 -or
        $FrameMtrandSyncEndUpdate -lt $FrameMtrandSyncStartUpdate -or
        $StopAfterUpdate -ge $FrameMtrandSyncStartUpdate -or
        $FrameMtrandSyncMaximumDraws -lt 0 -or
        $FrameMtrandSyncTimeoutSeconds -le 0
    )
) {
    throw (
        'frame MTRand sync requires a valid range strictly after the ' +
        'DMO trace stop update'
    )
}
if ($frameMtrandSyncRequested -and $postGlobalCallTraceRequested) {
    throw 'frame MTRand sync and global call tracing are mutually exclusive'
}
$gameplayMtrandSyncRequested = -not (
    [string]::IsNullOrWhiteSpace($GameplayMtrandOracle)
)
if (
    $gameplayMtrandSyncRequested -and
    (
        $GameplayMtrandMaximumDraws -lt 0 -or
        $GameplayMtrandSyncTimeoutSeconds -le 0
    )
) {
    throw 'gameplay MTRand synchronization options are invalid'
}
if (
    $gameplayMtrandSyncRequested -and
    ($frameMtrandSyncRequested -or $postGlobalCallTraceRequested)
) {
    throw (
        'gameplay MTRand synchronization is mutually exclusive with ' +
        'the other post-trace debuggers'
    )
}
$globalMtrandCallSyncRequested = -not (
    [string]::IsNullOrWhiteSpace($GlobalMtrandCallOracle)
)
if (
    $globalMtrandCallSyncRequested -and
    (
        $GlobalMtrandCallStartAfterUpdate -lt $StopAfterUpdate -or
        (
            $GlobalMtrandCallEndAtUpdate -ge 0 -and
            $GlobalMtrandCallEndAtUpdate -le
                $GlobalMtrandCallStartAfterUpdate
        ) -or
        $GlobalMtrandCallMaximumDraws -lt 0 -or
        $GlobalMtrandCallSyncTimeoutSeconds -le 0
    )
) {
    throw 'global MTRand call synchronization options are invalid'
}
if (
    $WaitForNaturalRuntimeExit -and
    (
        -not (
            $globalMtrandCallSyncRequested -or
            $globalCallRestoreRequested -or
            $threadCrtCallRestoreRequested
        ) -or
        $RuntimeExitTimeoutSeconds -le 0 -or
        $MinimumWinScore -lt 0 -or
        (
            $ExpectedOutcome -eq 'natural_win' -and
            $MinimumWinScore -le 0
        )
    )
) {
    throw (
        'full runtime-exit validation requires a bounded source oracle ' +
        'and positive validation bounds'
    )
}
if ($HeadlessFullReplay -and -not $WaitForNaturalRuntimeExit) {
    throw 'headless full replay requires runtime-exit validation'
}
if (
    $RequireSourceBoundGameplayParity -and
    (
        -not $WaitForNaturalRuntimeExit -or
        -not $hardwareGlobalCallTraceRequested -or
        -not $globalCallRestoreRequested -or
        -not $threadCrtCallRestoreRequested -or
        $GlobalCallRestoreTrace -ne $ThreadCrtCallRestoreTrace -or
        $GlobalCallRestoreRecordingReport -ne
            $ThreadCrtCallRestoreRecordingReport -or
        $GlobalCallRestoreSourceOrder -ne
            $ThreadCrtCallRestoreSourceOrder -or
        $GlobalCallRestoreFrameworkUpdate -ne
            $ThreadCrtCallRestoreFrameworkUpdate -or
        $GlobalCallRestoreCaller -ne $ThreadCrtCallRestoreCaller
    )
) {
    throw 'source-bound gameplay parity options are incomplete'
}
if (
    $globalMtrandCallSyncRequested -and
    (
        $gameplayMtrandSyncRequested -or
        $frameMtrandSyncRequested -or
        $postGlobalCallTraceRequested
    )
) {
    throw (
        'global MTRand call synchronization is mutually exclusive ' +
        'with the other post-trace debuggers'
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

function Wait-OwnedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [int]$ProcessId,
        [Parameter(Mandatory = $true)]
        [int]$TimeoutSeconds,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (-not (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)) {
            return
        }
        Start-Sleep -Milliseconds 200
    }
    throw "$Label process timeout"
}

Assert-ExistingFile -LiteralPath $Dmo -Label 'DMO'
Assert-ExistingFile -LiteralPath $ReferenceRngLog -Label 'reference RNG log'
Assert-ExistingFile -LiteralPath $Python -Label 'Python'
Assert-ExistingFile -LiteralPath $Runtime -Label 'direct runtime'
Assert-ExistingFile -LiteralPath $RecordPreSnapshot -Label 'record prestate'

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

$stateTool = Join-Path $ProjectRoot 'tools\pc_state_transaction.py'
$traceTool = Join-Path $ProjectRoot 'tools\trace_popcap_demo_commands.py'
$monitorTool = Join-Path $ProjectRoot 'tools\monitor_live_zuma_rng.py'
$globalCallTraceTool = Join-Path (
    Join-Path $ProjectRoot 'tools'
) 'trace_popcap_global_rng_calls.py'
$hardwareGlobalCallTraceTool = Join-Path (
    Join-Path $ProjectRoot 'tools'
) 'trace_popcap_gameplay_mtrand.py'
$frameMtrandSyncTool = Join-Path (
    Join-Path $ProjectRoot 'tools'
) 'synchronize_popcap_frame_mtrand.py'
$gameplayMtrandSyncTool = Join-Path (
    Join-Path $ProjectRoot 'tools'
) 'synchronize_popcap_gameplay_mtrand.py'
$globalMtrandCallSyncTool = Join-Path (
    Join-Path $ProjectRoot 'tools'
) 'synchronize_popcap_global_mtrand_calls.py'
$fullReplayMonitorTool = Join-Path (
    Join-Path $ProjectRoot 'tools'
) 'monitor_existing_popcap_replay.py'
$sourceBoundParityTool = Join-Path (
    Join-Path $ProjectRoot 'tools'
) 'compare_source_bound_popcap_replay.py'
Assert-ExistingFile -LiteralPath $stateTool -Label 'state transaction tool'
Assert-ExistingFile -LiteralPath $traceTool -Label 'trace tool'
Assert-ExistingFile -LiteralPath $monitorTool -Label 'RNG monitor'
if ($globalCallTraceRequested) {
    Assert-ExistingFile `
        -LiteralPath $globalCallTraceTool `
        -Label 'global RNG call trace tool'
}
if ($hardwareGlobalCallTraceRequested) {
    Assert-ExistingFile `
        -LiteralPath $hardwareGlobalCallTraceTool `
        -Label 'hardware global RNG call trace tool'
}
if ($threadCrtCallRestoreRequested) {
    Assert-ExistingFile `
        -LiteralPath $ThreadCrtCallRestoreTrace `
        -Label 'source-bound thread CRT trace'
    Assert-ExistingFile `
        -LiteralPath $ThreadCrtCallRestoreRecordingReport `
        -Label 'source-bound retail recording report'
}
if ($globalCallRestoreRequested -and $hardwareGlobalCallTraceRequested) {
    Assert-ExistingFile `
        -LiteralPath $GlobalCallRestoreTrace `
        -Label 'source-bound global MTRand trace'
    Assert-ExistingFile `
        -LiteralPath $GlobalCallRestoreRecordingReport `
        -Label 'source-bound global MTRand recording report'
}
if ($frameMtrandSyncRequested) {
    Assert-ExistingFile `
        -LiteralPath $frameMtrandSyncTool `
        -Label 'frame MTRand synchronizer'
}
if ($gameplayMtrandSyncRequested) {
    Assert-ExistingFile `
        -LiteralPath $GameplayMtrandOracle `
        -Label 'gameplay MTRand oracle'
    Assert-ExistingFile `
        -LiteralPath $gameplayMtrandSyncTool `
        -Label 'gameplay MTRand synchronizer'
}
if ($globalMtrandCallSyncRequested) {
    Assert-ExistingFile `
        -LiteralPath $GlobalMtrandCallOracle `
        -Label 'global MTRand call oracle'
    Assert-ExistingFile `
        -LiteralPath $globalMtrandCallSyncTool `
        -Label 'global MTRand call synchronizer'
}
if ($WaitForNaturalRuntimeExit) {
    Assert-ExistingFile `
        -LiteralPath $fullReplayMonitorTool `
        -Label 'full replay monitor'
}
if ($RequireSourceBoundGameplayParity) {
    Assert-ExistingFile `
        -LiteralPath $sourceBoundParityTool `
        -Label 'source-bound gameplay parity tool'
}

$nonce = [guid]::NewGuid().ToString('N')
$env:PYTHONPATH = "$(Join-Path $ProjectRoot 'src');$ProjectRoot"
$traceProcess = $null
$monitorProcess = $null
$globalCallTraceProcess = $null
$globalCallTraceOutput = $null
$globalCallTraceReady = $null
$globalCallTraceStop = $null
$hardwareGlobalCallTraceProcess = $null
$hardwareGlobalCallTraceOutput = $null
$hardwareGlobalCallTraceReady = $null
$hardwareGlobalCallTraceStop = $null
$frameMtrandSyncProcess = $null
$frameMtrandSyncOutput = $null
$frameMtrandSyncReady = $null
$frameMtrandSyncStop = $null
$gameplayMtrandSyncProcess = $null
$gameplayMtrandSyncOutput = $null
$gameplayMtrandSyncReady = $null
$gameplayMtrandSyncStop = $null
$globalMtrandCallSyncProcess = $null
$globalMtrandCallSyncOutput = $null
$globalMtrandCallSyncReady = $null
$globalMtrandCallSyncStop = $null
$fullReplayMonitorProcess = $null
$fullReplayMonitorOutput = $null
$sourceBoundParityOutput = $null
$runError = $null
$cleanupError = $null

New-Item -ItemType Directory -Path $OutputRoot | Out-Null

try {
    & $Python $stateTool snapshot `
        --session-nonce $nonce `
        --phase host-pre `
        --output (Join-Path $OutputRoot 'host-pre.json') | Out-Null

    & $Python $stateTool restore --snapshot $RecordPreSnapshot | Out-Null

    & $Python $stateTool snapshot `
        --session-nonce $nonce `
        --phase applied-record-pre `
        --output (Join-Path $OutputRoot 'applied-record-pre.json') | Out-Null

    $recordRoot = (
        Get-Content -Raw -LiteralPath $RecordPreSnapshot | ConvertFrom-Json
    ).state_root
    $appliedRoot = (
        Get-Content -Raw -LiteralPath (
            Join-Path $OutputRoot 'applied-record-pre.json'
        ) | ConvertFrom-Json
    ).state_root
    if ($recordRoot -ne $appliedRoot) {
        throw "recording prestate mismatch: $appliedRoot"
    }

    # Start-Process joins ArgumentList with spaces on Windows. Preserve the
    # one retail asset path containing spaces by passing literal quote marks.
    $quotedChangedir = '"' + $Changedir + '"'
    $traceArguments = @(
        $traceTool,
        '--dmo', $Dmo,
        '--runtime-exe', $Runtime,
        '--direct-runtime-exe', $Runtime,
        '--changedir', $quotedChangedir,
        '--crt-rand-seed', "$StartupCrtSeed",
        '--startup-seed-transport', $StartupSeedTransport,
        '--maximum-hits', "$MaximumHits",
        '--trace-timeout', "$TraceTimeoutSeconds",
        '--attach-at-update', "$AttachAtUpdate",
        '--broker-service-blocks',
        '--service-wait-timeout', "$ServiceWaitTimeoutSeconds",
        '--quiet-nonservice',
        '--stop-after-update', "$StopAfterUpdate",
        '--progress-every-updates', '500',
        '--result-json', (Join-Path $OutputRoot 'result.json')
    )
    if (-not $DisableBoundaryRngRestores) {
        $traceArguments += @(
            '--global-mtrand-restore-log', $ReferenceRngLog,
            '--global-mtrand-restore-source-update', "$GlobalSourceUpdate",
            '--global-mtrand-expected-before-log', $ReferenceRngLog,
            '--global-mtrand-expected-before-source-update',
            "$GlobalSourceUpdate",
            '--global-mtrand-restore-seed', "$GlobalMtrandSeed",
            '--global-mtrand-restore-rewind-draws',
            "$GlobalRestoreRewindDraws",
            '--global-mtrand-restore-maximum-draws', '10000',
            '--global-mtrand-restore-command-order',
            "$GlobalRestoreCommandOrder",
            '--global-mtrand-allow-dynamic-before',
            '--thread-crt-restore-log', $ReferenceRngLog,
            '--thread-crt-restore-source-update', "$ThreadCrtSourceUpdate",
            '--thread-crt-expected-before-log', $ReferenceRngLog,
            '--thread-crt-expected-before-source-update',
            "$ThreadCrtSourceUpdate",
            '--thread-crt-restore-command-order',
            "$ThreadCrtRestoreCommandOrder",
            '--thread-crt-restore-rewind-draws',
            "$ThreadCrtRestoreRewindDraws",
            '--thread-crt-allow-dynamic-before'
        )
    }
    if ($StartupTraceHandoff) {
        $traceArguments += '--startup-trace-handoff'
    }
    if ($StartupPriorityBiasUntilUpdate -gt 0) {
        $traceArguments += @(
            '--startup-priority-bias-until-update',
            "$StartupPriorityBiasUntilUpdate"
        )
    }
    if ($AttachAtUpdate -eq 0) {
        $traceArguments += '--allow-pre-stream-commands'
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
    if ($globalMtrandCallSyncRequested) {
        $traceArguments += '--suspend-main-thread-on-stop'
    }
    $traceProcess = Start-Process `
        -FilePath $Python `
        -ArgumentList $traceArguments `
        -RedirectStandardOutput (Join-Path $OutputRoot 'trace.log') `
        -RedirectStandardError (Join-Path $OutputRoot 'trace.err.log') `
        -PassThru `
        -WindowStyle Hidden

    $launchDeadline = [DateTime]::UtcNow.AddSeconds(45)
    $runtimeProcessId = $null
    $mainThreadId = $null
    while ([DateTime]::UtcNow -lt $launchDeadline) {
        $direct = Get-ExactRuntimeProcesses | Select-Object -First 1
        if ($null -ne $direct) {
            $runtimeProcessId = [int]$direct.ProcessId
        }

        $traceLog = Join-Path $OutputRoot 'trace.log'
        if (Test-Path -LiteralPath $traceLog) {
            $match = Select-String `
                -LiteralPath $traceLog `
                -Pattern 'startup_rng .* tid=(\d+)' |
                Select-Object -First 1
            if ($null -ne $match -and $match.Matches.Count -gt 0) {
                $mainThreadId = [int]$match.Matches[0].Groups[1].Value
            }
        }
        if ($null -ne $runtimeProcessId -and $null -ne $mainThreadId) {
            break
        }
        if (-not (Get-Process -Id $traceProcess.Id -ErrorAction SilentlyContinue)) {
            break
        }
        Start-Sleep -Milliseconds 100
    }
    if ($null -eq $runtimeProcessId) {
        throw 'direct runtime pid was not observed'
    }
    if ($null -eq $mainThreadId) {
        throw 'startup main thread id was not observed'
    }

    $monitorArguments = @(
        $monitorTool,
        '--pid', "$runtimeProcessId",
        '--output', (Join-Path $OutputRoot 'rng.ndjson'),
        '--interval-seconds', "$MonitorIntervalSeconds",
        '--maximum-seconds', "$MonitorTimeoutSeconds",
        '--stop-after-native-game-time', "$StopAfterNativeGameTime"
    )
    if (-not $DisableBoundaryRngRestores) {
        $monitorArguments += @('--thread-id', "$mainThreadId")
    }
    $monitorProcess = Start-Process `
        -FilePath $Python `
        -ArgumentList $monitorArguments `
        -RedirectStandardOutput (Join-Path $OutputRoot 'rng-monitor.log') `
        -RedirectStandardError (Join-Path $OutputRoot 'rng-monitor.err.log') `
        -PassThru `
        -WindowStyle Hidden

    if ($WaitForNaturalRuntimeExit) {
        $fullReplayMonitorOutput = Join-Path (
            $OutputRoot
        ) 'full-replay.json'
        $fullReplaySnapshotRoot = Join-Path (
            $OutputRoot
        ) 'ui-snapshots'
        $fullReplayMonitorArguments = @(
            $fullReplayMonitorTool,
            '--pid', "$runtimeProcessId",
            '--executable', $Runtime,
            '--dmo', $Dmo,
            '--output', $fullReplayMonitorOutput,
            '--snapshot-root', $fullReplaySnapshotRoot,
            '--minimum-win-score', "$MinimumWinScore",
            '--expected-outcome', $ExpectedOutcome,
            '--timeout', "$RuntimeExitTimeoutSeconds",
            '--sample-interval', '0.001',
            '--strict-trace-result', (
                Join-Path $OutputRoot 'result.json'
            )
        )
        if (-not $HeadlessFullReplay) {
            $fullReplayMonitorArguments += @(
                '--snapshot', '5380:score_continue',
                '--snapshot', '6400:next_level_menu',
                '--snapshot', '6620:pause_main_menu',
                '--snapshot', '6760:return_to_main_confirm',
                '--snapshot', '6920:main_quit',
                '--snapshot', '7080:quit_confirm'
            )
        }
        if ($RequireSourceBoundGameplayParity) {
            $fullReplayMonitorArguments += @(
                '--defer-natural-outcome-to-source-bound-parity'
            )
        }
        $fullReplayMonitorProcess = Start-Process `
            -FilePath $Python `
            -ArgumentList $fullReplayMonitorArguments `
            -RedirectStandardOutput (
                Join-Path $OutputRoot 'full-replay.log'
            ) `
            -RedirectStandardError (
                Join-Path $OutputRoot 'full-replay.err.log'
            ) `
            -PassThru `
            -WindowStyle Hidden
    }

    if ($postGlobalCallTraceRequested -and $AttachAtUpdate -gt 0) {
        $traceDebuggerReady = $false
        $traceDebuggerDeadline = [DateTime]::UtcNow.AddSeconds(45)
        $traceLogPath = Join-Path $OutputRoot 'trace.log'
        while ([DateTime]::UtcNow -lt $traceDebuggerDeadline) {
            if (
                (Test-Path -LiteralPath $traceLogPath) -and
                $null -ne (
                    Select-String `
                        -LiteralPath $traceLogPath `
                        -Pattern '^armed pid=' |
                        Select-Object -First 1
                )
            ) {
                $traceDebuggerReady = $true
                break
            }
            if (
                -not (
                    Get-Process `
                        -Id $traceProcess.Id `
                        -ErrorAction SilentlyContinue
                )
            ) {
                break
            }
            Start-Sleep -Milliseconds 50
        }
        if (-not $traceDebuggerReady) {
            throw 'late DMO trace did not establish debugger ownership'
        }
    }

    if ($globalCallTraceRequested) {
        $globalCallTraceOutput = Join-Path (
            $OutputRoot
        ) 'global-rng-calls.json'
        $globalCallTraceReady = Join-Path (
            $OutputRoot
        ) 'global-rng-calls.ready.json'
        $globalCallTraceStop = Join-Path (
            $OutputRoot
        ) 'global-rng-calls.stop'
        $globalCallTraceArguments = @(
            $globalCallTraceTool,
            '--pid', "$runtimeProcessId",
            '--executable', $Runtime,
            '--output', $globalCallTraceOutput,
            '--ready', $globalCallTraceReady,
            '--stop', $globalCallTraceStop,
            '--start-update', "$GlobalCallTraceStartUpdate",
            '--end-update', "$GlobalCallTraceEndUpdate",
            '--attach-timeout', "$TraceTimeoutSeconds",
            '--timeout', "$GlobalCallTraceTimeoutSeconds"
        )
        if ($globalCallRestoreRequested) {
            $globalCallTraceArguments += @(
                '--restore-log', $ReferenceRngLog,
                '--restore-source-update',
                "$GlobalCallRestoreSourceUpdate",
                '--restore-seed', "$GlobalMtrandSeed",
                '--restore-rewind-draws',
                "$GlobalCallRestoreRewindDraws",
                '--restore-maximum-draws', '10000',
                '--restore-framework-update',
                "$GlobalCallRestoreFrameworkUpdate",
                '--restore-caller',
                ('0x{0:x8}' -f $GlobalCallRestoreCaller)
            )
        }
        if ($threadCrtCallRestoreRequested) {
            $globalCallTraceArguments += @(
                '--thread-crt-restore-trace',
                $ThreadCrtCallRestoreTrace,
                '--thread-crt-restore-recording-report',
                $ThreadCrtCallRestoreRecordingReport,
                '--thread-crt-restore-source-order',
                "$ThreadCrtCallRestoreSourceOrder",
                '--thread-crt-restore-framework-update',
                "$ThreadCrtCallRestoreFrameworkUpdate",
                '--thread-crt-restore-caller',
                ('0x{0:x8}' -f $ThreadCrtCallRestoreCaller)
            )
        }
        # Launch while the DMO tracer still owns the debug port.  The call
        # tracer performs a bounded ACCESS_DENIED retry and acquires the port
        # immediately after the first tracer detaches, avoiding an untraced
        # handoff window in the running game.
        $globalCallTraceProcess = Start-Process `
            -FilePath $Python `
            -ArgumentList $globalCallTraceArguments `
            -RedirectStandardOutput (
                Join-Path $OutputRoot 'global-rng-calls.log'
            ) `
            -RedirectStandardError (
                Join-Path $OutputRoot 'global-rng-calls.err.log'
            ) `
            -PassThru `
            -WindowStyle Hidden
    }
    if ($hardwareGlobalCallTraceRequested) {
        if ($HardwareGlobalCallTraceLaunchAfterUpdate -ge 0) {
            $launchAfterDeadline = [DateTime]::UtcNow.AddSeconds(
                $TraceTimeoutSeconds
            )
            $launchAfterObserved = $false
            $traceLogPath = Join-Path $OutputRoot 'trace.log'
            while ([DateTime]::UtcNow -lt $launchAfterDeadline) {
                if (Test-Path -LiteralPath $traceLogPath) {
                    $progressRows = Select-String `
                        -LiteralPath $traceLogPath `
                        -Pattern '^trace_progress .* update=(\d+)'
                    foreach ($progressRow in @($progressRows)) {
                        if (
                            $progressRow.Matches.Count -gt 0 -and
                            [int](
                                $progressRow.Matches[0].Groups[1].Value
                            ) -ge $HardwareGlobalCallTraceLaunchAfterUpdate
                        ) {
                            $launchAfterObserved = $true
                            break
                        }
                    }
                }
                if ($launchAfterObserved) {
                    break
                }
                if (
                    -not (
                        Get-Process `
                            -Id $traceProcess.Id `
                            -ErrorAction SilentlyContinue
                    )
                ) {
                    break
                }
                Start-Sleep -Milliseconds 50
            }
            if (-not $launchAfterObserved) {
                throw (
                    'strict DMO trace did not reach the requested hardware ' +
                    'tracer launch boundary'
                )
            }
        }
        $hardwareGlobalCallTraceOutput = Join-Path (
            $OutputRoot
        ) 'global-mtrand-hardware-calls.json'
        $hardwareGlobalCallTraceReady = Join-Path (
            $OutputRoot
        ) 'global-mtrand-hardware-calls.ready.json'
        $hardwareGlobalCallTraceStop = Join-Path (
            $OutputRoot
        ) 'global-mtrand-hardware-calls.stop'
        $hardwareGlobalCallTraceArguments = @(
            $hardwareGlobalCallTraceTool,
            '--pid', "$runtimeProcessId",
            '--main-thread-id', "$mainThreadId",
            '--executable', $Runtime,
            '--output', $hardwareGlobalCallTraceOutput,
            '--ready', $hardwareGlobalCallTraceReady,
            '--stop', $hardwareGlobalCallTraceStop,
            '--breakpoint-kind', 'global_wrapper_entry',
            '--attach-timeout', "$TraceTimeoutSeconds",
            '--timeout', "$HardwareGlobalCallTraceTimeoutSeconds",
            '--maximum-calls', "$HardwareGlobalCallTraceMaximumCalls"
        )
        if ($globalCallRestoreRequested) {
            $hardwareGlobalCallTraceArguments += @(
                '--global-mtrand-restore-monitor', $ReferenceRngLog,
                '--global-mtrand-restore-trace',
                $GlobalCallRestoreTrace,
                '--global-mtrand-restore-recording-report',
                $GlobalCallRestoreRecordingReport,
                '--global-mtrand-restore-monitor-update',
                "$GlobalCallRestoreSourceUpdate",
                '--global-mtrand-restore-seed', "$GlobalMtrandSeed",
                '--global-mtrand-restore-rewind-draws',
                "$GlobalCallRestoreRewindDraws",
                '--global-mtrand-restore-maximum-draws', '10000',
                '--global-mtrand-restore-source-order',
                "$GlobalCallRestoreSourceOrder",
                '--global-mtrand-restore-framework-update',
                "$GlobalCallRestoreFrameworkUpdate",
                '--global-mtrand-restore-caller',
                ('0x{0:x8}' -f $GlobalCallRestoreCaller)
            )
        }
        if ($threadCrtCallRestoreRequested) {
            $hardwareGlobalCallTraceArguments += @(
                '--thread-crt-restore-trace',
                $ThreadCrtCallRestoreTrace,
                '--thread-crt-restore-recording-report',
                $ThreadCrtCallRestoreRecordingReport,
                '--thread-crt-restore-source-order',
                "$ThreadCrtCallRestoreSourceOrder",
                '--thread-crt-restore-framework-update',
                "$ThreadCrtCallRestoreFrameworkUpdate",
                '--thread-crt-restore-caller',
                ('0x{0:x8}' -f $ThreadCrtCallRestoreCaller)
            )
        }
        # Launch before the strict DMO tracer releases the debug port.  The
        # hardware tracer waits for that ownership handoff and then observes
        # the same main-thread boundary used by the natural source recording.
        $hardwareGlobalCallTraceProcess = Start-Process `
            -FilePath $Python `
            -ArgumentList $hardwareGlobalCallTraceArguments `
            -RedirectStandardOutput (
                Join-Path $OutputRoot 'global-mtrand-hardware-calls.log'
            ) `
            -RedirectStandardError (
                Join-Path $OutputRoot 'global-mtrand-hardware-calls.err.log'
            ) `
            -PassThru `
            -WindowStyle Hidden
    }
    if ($frameMtrandSyncRequested) {
        $frameMtrandSyncOutput = Join-Path (
            $OutputRoot
        ) 'frame-mtrand-sync.json'
        $frameMtrandSyncReady = Join-Path (
            $OutputRoot
        ) 'frame-mtrand-sync.ready.json'
        $frameMtrandSyncStop = Join-Path (
            $OutputRoot
        ) 'frame-mtrand-sync.stop'
        $frameMtrandSyncArguments = @(
            $frameMtrandSyncTool,
            '--pid', "$runtimeProcessId",
            '--main-thread-id', "$mainThreadId",
            '--executable', $Runtime,
            '--reference-log', $ReferenceRngLog,
            '--seed', "$GlobalMtrandSeed",
            '--start-update', "$FrameMtrandSyncStartUpdate",
            '--end-update', "$FrameMtrandSyncEndUpdate",
            '--maximum-draws', "$FrameMtrandSyncMaximumDraws",
            '--output', $frameMtrandSyncOutput,
            '--ready', $frameMtrandSyncReady,
            '--stop', $frameMtrandSyncStop,
            '--attach-timeout', "$TraceTimeoutSeconds",
            '--timeout', "$FrameMtrandSyncTimeoutSeconds"
        )
        # Start during DMO tracing so the bounded debugger-handoff retry
        # acquires the target without an uncontrolled gameplay interval.
        $frameMtrandSyncProcess = Start-Process `
            -FilePath $Python `
            -ArgumentList $frameMtrandSyncArguments `
            -RedirectStandardOutput (
                Join-Path $OutputRoot 'frame-mtrand-sync.log'
            ) `
            -RedirectStandardError (
                Join-Path $OutputRoot 'frame-mtrand-sync.err.log'
            ) `
            -PassThru `
            -WindowStyle Hidden
    }
    if ($gameplayMtrandSyncRequested) {
        $gameplayMtrandSyncOutput = Join-Path (
            $OutputRoot
        ) 'gameplay-mtrand-sync.json'
        $gameplayMtrandSyncReady = Join-Path (
            $OutputRoot
        ) 'gameplay-mtrand-sync.ready.json'
        $gameplayMtrandSyncStop = Join-Path (
            $OutputRoot
        ) 'gameplay-mtrand-sync.stop'
        $gameplayMtrandSyncArguments = @(
            $gameplayMtrandSyncTool,
            '--pid', "$runtimeProcessId",
            '--main-thread-id', "$mainThreadId",
            '--executable', $Runtime,
            '--oracle', $GameplayMtrandOracle,
            '--seed', "$GlobalMtrandSeed",
            '--maximum-draws', "$GameplayMtrandMaximumDraws",
            '--output', $gameplayMtrandSyncOutput,
            '--ready', $gameplayMtrandSyncReady,
            '--stop', $gameplayMtrandSyncStop,
            '--attach-timeout', "$TraceTimeoutSeconds",
            '--timeout', "$GameplayMtrandSyncTimeoutSeconds"
        )
        # Launch during startup tracing to eliminate the debugger-handoff
        # window before the first gameplay call.
        $gameplayMtrandSyncProcess = Start-Process `
            -FilePath $Python `
            -ArgumentList $gameplayMtrandSyncArguments `
            -RedirectStandardOutput (
                Join-Path $OutputRoot 'gameplay-mtrand-sync.log'
            ) `
            -RedirectStandardError (
                Join-Path $OutputRoot 'gameplay-mtrand-sync.err.log'
            ) `
            -PassThru `
            -WindowStyle Hidden
    }
    if ($globalMtrandCallSyncRequested) {
        $globalMtrandCallSyncOutput = Join-Path (
            $OutputRoot
        ) 'global-mtrand-call-sync.json'
        $globalMtrandCallSyncReady = Join-Path (
            $OutputRoot
        ) 'global-mtrand-call-sync.ready.json'
        $globalMtrandCallSyncStop = Join-Path (
            $OutputRoot
        ) 'global-mtrand-call-sync.stop'
        $globalMtrandCallSyncArguments = @(
            $globalMtrandCallSyncTool,
            '--pid', "$runtimeProcessId",
            '--main-thread-id', "$mainThreadId",
            '--executable', $Runtime,
            '--oracle', $GlobalMtrandCallOracle,
            '--seed', "$GlobalMtrandSeed",
            '--start-after-update',
            "$GlobalMtrandCallStartAfterUpdate",
            '--maximum-draws', "$GlobalMtrandCallMaximumDraws",
            '--output', $globalMtrandCallSyncOutput,
            '--ready', $globalMtrandCallSyncReady,
            '--stop', $globalMtrandCallSyncStop,
            '--resume-main-thread-on-ready',
            '--attach-timeout', "$TraceTimeoutSeconds",
            '--timeout', "$GlobalMtrandCallSyncTimeoutSeconds"
        )
        if ($GlobalMtrandCallEndAtUpdate -ge 0) {
            $globalMtrandCallSyncArguments += @(
                '--end-at-update',
                "$GlobalMtrandCallEndAtUpdate"
            )
        }
        $globalMtrandCallSyncProcess = Start-Process `
            -FilePath $Python `
            -ArgumentList $globalMtrandCallSyncArguments `
            -RedirectStandardOutput (
                Join-Path $OutputRoot 'global-mtrand-call-sync.log'
            ) `
            -RedirectStandardError (
                Join-Path $OutputRoot 'global-mtrand-call-sync.err.log'
            ) `
            -PassThru `
            -WindowStyle Hidden
    }

    Wait-OwnedProcess `
        -ProcessId $traceProcess.Id `
        -TimeoutSeconds $TraceTimeoutSeconds `
        -Label 'trace'

    $resultPath = Join-Path $OutputRoot 'result.json'
    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw 'trace result was not created'
    }
    $traceResult = Get-Content -Raw -LiteralPath $resultPath |
        ConvertFrom-Json
    if (
        [int]$traceResult.result.failure_count -ne 0 -or
        -not [bool]$traceResult.result.stopped_at_update -or
        [int]$traceResult.result.last_update -lt $StopAfterUpdate
    ) {
        throw 'trace structured result did not pass'
    }
    if (
        $globalMtrandCallSyncRequested -and
        (
            -not [bool](
                $traceResult.result.handoff_main_thread_suspended
            ) -or
            [int]$traceResult.result.handoff_main_thread_id -ne
                $mainThreadId -or
            [int](
                $traceResult.result.handoff_previous_suspend_count
            ) -ne 0
        )
    ) {
        throw 'trace did not establish the frozen debugger handoff'
    }

    if ($hardwareGlobalCallTraceRequested) {
        $readyDeadline = [DateTime]::UtcNow.AddSeconds(30)
        while (
            [DateTime]::UtcNow -lt $readyDeadline -and
            -not (
                Test-Path -LiteralPath $hardwareGlobalCallTraceReady
            ) -and
            (
                Get-Process `
                    -Id $hardwareGlobalCallTraceProcess.Id `
                    -ErrorAction SilentlyContinue
            )
        ) {
            Start-Sleep -Milliseconds 50
        }
        if (-not (Test-Path -LiteralPath $hardwareGlobalCallTraceReady)) {
            throw 'hardware global RNG call trace did not become ready'
        }

        if (-not $WaitForNaturalRuntimeExit) {
            $targetDeadline = [DateTime]::UtcNow.AddSeconds(
                $HardwareGlobalCallTraceTimeoutSeconds
            )
            $targetPassed = $false
            while ([DateTime]::UtcNow -lt $targetDeadline) {
                $rngPath = Join-Path $OutputRoot 'rng.ndjson'
                if (Test-Path -LiteralPath $rngPath) {
                    try {
                        $lastRngRow = Get-Content `
                            -LiteralPath $rngPath `
                            -Tail 1 |
                            ConvertFrom-Json
                        if (
                            $lastRngRow.type -eq 'rng' -and
                            [int]$lastRngRow.native_game_time -ge
                                $StopAfterNativeGameTime
                        ) {
                            $targetPassed = $true
                            break
                        }
                    }
                    catch {
                        # The monitor may be between an append and newline.
                    }
                }
                if (
                    -not (
                        Get-Process `
                            -Id $hardwareGlobalCallTraceProcess.Id `
                            -ErrorAction SilentlyContinue
                    )
                ) {
                    break
                }
                Start-Sleep -Milliseconds 50
            }
            if (
                $targetPassed -and
                (
                    Get-Process `
                        -Id $hardwareGlobalCallTraceProcess.Id `
                        -ErrorAction SilentlyContinue
                )
            ) {
                New-Item `
                    -ItemType File `
                    -Path $hardwareGlobalCallTraceStop |
                    Out-Null
            }
        }

        Wait-OwnedProcess `
            -ProcessId $hardwareGlobalCallTraceProcess.Id `
            -TimeoutSeconds $HardwareGlobalCallTraceTimeoutSeconds `
            -Label 'hardware global RNG call trace'

        if (-not (Test-Path -LiteralPath $hardwareGlobalCallTraceOutput)) {
            throw 'hardware global RNG call trace result was not created'
        }
        $hardwareGlobalCallTraceResult = Get-Content `
            -Raw `
            -LiteralPath $hardwareGlobalCallTraceOutput |
            ConvertFrom-Json
        $hardwareThreadCrtRestore = (
            $hardwareGlobalCallTraceResult.thread_crt_call_site_restore
        )
        $hardwareGlobalMtrandRestore = (
            $hardwareGlobalCallTraceResult.global_mtrand_call_site_restore
        )
        $expectedHardwareProcessWrites = 0
        if ($null -ne $hardwareGlobalMtrandRestore) {
            $expectedHardwareProcessWrites += [int](
                $hardwareGlobalMtrandRestore.bytes_written
            )
        }
        if ($null -ne $hardwareThreadCrtRestore) {
            $expectedHardwareProcessWrites += [int](
                $hardwareThreadCrtRestore.bytes_written
            )
        }
        if (
            $hardwareGlobalCallTraceResult.status -ne 'PASS' -or
            [int]$hardwareGlobalCallTraceResult.call_count -le 0 -or
            $hardwareGlobalCallTraceResult.breakpoint_kind -ne
                'global_wrapper_entry' -or
            -not [bool](
                $hardwareGlobalCallTraceResult.hardware_breakpoint_restored
            ) -or
            [bool]$hardwareGlobalCallTraceResult.persistent_file_modified -or
            (
                $globalCallRestoreRequested -and
                $null -eq $hardwareGlobalMtrandRestore
            ) -or
            (
                $threadCrtCallRestoreRequested -and
                $null -eq $hardwareThreadCrtRestore
            ) -or
            (
                $null -ne $hardwareGlobalMtrandRestore -and
                (
                    -not [bool](
                        $hardwareGlobalMtrandRestore.writeback_verified
                    )
                )
            ) -or
            (
                $null -ne $hardwareThreadCrtRestore -and
                (
                    -not [bool](
                        $hardwareThreadCrtRestore.writeback_verified
                    )
                )
            ) -or
            [int](
                $hardwareGlobalCallTraceResult.process_memory_writes
            ) -ne $expectedHardwareProcessWrites -or
            (
                $WaitForNaturalRuntimeExit -and
                (
                    $hardwareGlobalCallTraceResult.stop_reason -ne
                        'process_exit' -or
                    -not [bool]$hardwareGlobalCallTraceResult.exited -or
                    [int]$hardwareGlobalCallTraceResult.exit_code -ne 0
                )
            )
        ) {
            throw 'hardware global RNG call trace structured result did not pass'
        }
    }

    if ($globalCallTraceRequested) {
        $readyDeadline = [DateTime]::UtcNow.AddSeconds(30)
        while (
            [DateTime]::UtcNow -lt $readyDeadline -and
            -not (Test-Path -LiteralPath $globalCallTraceReady) -and
            (
                Get-Process `
                    -Id $globalCallTraceProcess.Id `
                    -ErrorAction SilentlyContinue
            )
        ) {
            Start-Sleep -Milliseconds 50
        }
        if (-not (Test-Path -LiteralPath $globalCallTraceReady)) {
            throw 'global RNG call trace did not become ready'
        }

        $rangeDeadline = [DateTime]::UtcNow.AddSeconds(
            $GlobalCallTraceTimeoutSeconds
        )
        $rangePassed = $false
        while ([DateTime]::UtcNow -lt $rangeDeadline) {
            $rngPath = Join-Path $OutputRoot 'rng.ndjson'
            if (Test-Path -LiteralPath $rngPath) {
                try {
                    $lastRngRow = Get-Content `
                        -LiteralPath $rngPath `
                        -Tail 1 |
                        ConvertFrom-Json
                    if (
                        $lastRngRow.type -eq 'rng' -and
                        [int]$lastRngRow.framework_update -gt
                            $GlobalCallTraceEndUpdate
                    ) {
                        $rangePassed = $true
                        break
                    }
                }
                catch {
                    # The monitor may be between one append and its newline.
                }
            }
            if (
                -not (
                    Get-Process `
                        -Id $globalCallTraceProcess.Id `
                        -ErrorAction SilentlyContinue
                )
            ) {
                break
            }
            Start-Sleep -Milliseconds 50
        }
        if ($rangePassed) {
            New-Item -ItemType File -Path $globalCallTraceStop | Out-Null
        }

        Wait-OwnedProcess `
            -ProcessId $globalCallTraceProcess.Id `
            -TimeoutSeconds $GlobalCallTraceTimeoutSeconds `
            -Label 'global RNG call trace'

        if (-not (Test-Path -LiteralPath $globalCallTraceOutput)) {
            throw 'global RNG call trace result was not created'
        }
        $globalCallTraceResult = Get-Content `
            -Raw `
            -LiteralPath $globalCallTraceOutput |
            ConvertFrom-Json
        if (
            $globalCallTraceResult.status -ne 'PASS' -or
            [int]$globalCallTraceResult.call_count -le 0 -or
            (
                $globalCallRestoreRequested -and
                $null -eq (
                    $globalCallTraceResult.global_mtrand_call_site_restore
                )
            ) -or
            (
                $threadCrtCallRestoreRequested -and
                $null -eq (
                    $globalCallTraceResult.thread_crt_call_site_restore
                )
            )
        ) {
            throw 'global RNG call trace structured result did not pass'
        }
    }
    if ($frameMtrandSyncRequested) {
        $readyDeadline = [DateTime]::UtcNow.AddSeconds(30)
        while (
            [DateTime]::UtcNow -lt $readyDeadline -and
            -not (Test-Path -LiteralPath $frameMtrandSyncReady) -and
            (
                Get-Process `
                    -Id $frameMtrandSyncProcess.Id `
                    -ErrorAction SilentlyContinue
            )
        ) {
            Start-Sleep -Milliseconds 50
        }
        if (-not (Test-Path -LiteralPath $frameMtrandSyncReady)) {
            throw 'frame MTRand synchronizer did not become ready'
        }

        $rangeDeadline = [DateTime]::UtcNow.AddSeconds(
            $FrameMtrandSyncTimeoutSeconds
        )
        $rangePassed = $false
        while ([DateTime]::UtcNow -lt $rangeDeadline) {
            $rngPath = Join-Path $OutputRoot 'rng.ndjson'
            if (Test-Path -LiteralPath $rngPath) {
                try {
                    $lastRngRow = Get-Content `
                        -LiteralPath $rngPath `
                        -Tail 1 |
                        ConvertFrom-Json
                    if (
                        $lastRngRow.type -eq 'rng' -and
                        [int]$lastRngRow.framework_update -gt
                            $FrameMtrandSyncEndUpdate
                    ) {
                        $rangePassed = $true
                        break
                    }
                }
                catch {
                    # The monitor may be between one append and its newline.
                }
            }
            if (
                -not (
                    Get-Process `
                        -Id $frameMtrandSyncProcess.Id `
                        -ErrorAction SilentlyContinue
                )
            ) {
                break
            }
            Start-Sleep -Milliseconds 50
        }
        if (
            $rangePassed -and
            (
                Get-Process `
                    -Id $frameMtrandSyncProcess.Id `
                    -ErrorAction SilentlyContinue
            )
        ) {
            New-Item -ItemType File -Path $frameMtrandSyncStop | Out-Null
        }

        Wait-OwnedProcess `
            -ProcessId $frameMtrandSyncProcess.Id `
            -TimeoutSeconds $FrameMtrandSyncTimeoutSeconds `
            -Label 'frame MTRand synchronizer'

        if (-not (Test-Path -LiteralPath $frameMtrandSyncOutput)) {
            throw 'frame MTRand synchronizer result was not created'
        }
        $frameMtrandSyncResult = Get-Content `
            -Raw `
            -LiteralPath $frameMtrandSyncOutput |
            ConvertFrom-Json
        if (
            $frameMtrandSyncResult.status -ne 'PASS' -or
            [int]$frameMtrandSyncResult.expected_hit_count -le 0 -or
            [int]$frameMtrandSyncResult.hit_count -ne
                [int]$frameMtrandSyncResult.expected_hit_count -or
            [int]$frameMtrandSyncResult.missing_hit_count -ne 0 -or
            @($frameMtrandSyncResult.unexpected_hits).Count -ne 0 -or
            @($frameMtrandSyncResult.duplicate_hits).Count -ne 0 -or
            -not [bool]$frameMtrandSyncResult.hardware_breakpoint_restored
        ) {
            throw 'frame MTRand synchronizer structured result did not pass'
        }
    }
    if ($gameplayMtrandSyncRequested) {
        $readyDeadline = [DateTime]::UtcNow.AddSeconds(30)
        while (
            [DateTime]::UtcNow -lt $readyDeadline -and
            -not (Test-Path -LiteralPath $gameplayMtrandSyncReady) -and
            (
                Get-Process `
                    -Id $gameplayMtrandSyncProcess.Id `
                    -ErrorAction SilentlyContinue
            )
        ) {
            Start-Sleep -Milliseconds 50
        }
        if (-not (Test-Path -LiteralPath $gameplayMtrandSyncReady)) {
            throw 'gameplay MTRand synchronizer did not become ready'
        }

        $targetDeadline = [DateTime]::UtcNow.AddSeconds(
            $GameplayMtrandSyncTimeoutSeconds
        )
        $targetPassed = $false
        while ([DateTime]::UtcNow -lt $targetDeadline) {
            $rngPath = Join-Path $OutputRoot 'rng.ndjson'
            if (Test-Path -LiteralPath $rngPath) {
                try {
                    $lastRngRow = Get-Content `
                        -LiteralPath $rngPath `
                        -Tail 1 |
                        ConvertFrom-Json
                    if (
                        $lastRngRow.type -eq 'rng' -and
                        [int]$lastRngRow.native_game_time -ge
                            $StopAfterNativeGameTime
                    ) {
                        $targetPassed = $true
                        break
                    }
                }
                catch {
                    # The monitor may be between one append and its newline.
                }
            }
            if (
                -not (
                    Get-Process `
                        -Id $gameplayMtrandSyncProcess.Id `
                        -ErrorAction SilentlyContinue
                )
            ) {
                break
            }
            Start-Sleep -Milliseconds 50
        }
        if (
            $targetPassed -and
            (
                Get-Process `
                    -Id $gameplayMtrandSyncProcess.Id `
                    -ErrorAction SilentlyContinue
            )
        ) {
            New-Item -ItemType File -Path $gameplayMtrandSyncStop |
                Out-Null
        }

        Wait-OwnedProcess `
            -ProcessId $gameplayMtrandSyncProcess.Id `
            -TimeoutSeconds $GameplayMtrandSyncTimeoutSeconds `
            -Label 'gameplay MTRand synchronizer'

        if (-not (Test-Path -LiteralPath $gameplayMtrandSyncOutput)) {
            throw 'gameplay MTRand synchronizer result was not created'
        }
        $gameplayMtrandSyncResult = Get-Content `
            -Raw `
            -LiteralPath $gameplayMtrandSyncOutput |
            ConvertFrom-Json
        if (
            $gameplayMtrandSyncResult.status -ne 'PASS' -or
            [int]$gameplayMtrandSyncResult.expected_hit_count -le 0 -or
            [int]$gameplayMtrandSyncResult.hit_count -ne
                [int]$gameplayMtrandSyncResult.expected_hit_count -or
            [int]$gameplayMtrandSyncResult.missing_hit_count -ne 0 -or
            $null -ne $gameplayMtrandSyncResult.unexpected_extra_call -or
            -not [bool](
                $gameplayMtrandSyncResult.hardware_breakpoint_restored
            )
        ) {
            throw (
                'gameplay MTRand synchronizer structured result ' +
                'did not pass'
            )
        }
    }
    if ($globalMtrandCallSyncRequested) {
        $readyDeadline = [DateTime]::UtcNow.AddSeconds(30)
        while (
            [DateTime]::UtcNow -lt $readyDeadline -and
            -not (Test-Path -LiteralPath $globalMtrandCallSyncReady) -and
            (
                Get-Process `
                    -Id $globalMtrandCallSyncProcess.Id `
                    -ErrorAction SilentlyContinue
            )
        ) {
            Start-Sleep -Milliseconds 50
        }
        if (-not (Test-Path -LiteralPath $globalMtrandCallSyncReady)) {
            throw 'global MTRand call synchronizer did not become ready'
        }

        Wait-OwnedProcess `
            -ProcessId $globalMtrandCallSyncProcess.Id `
            -TimeoutSeconds $GlobalMtrandCallSyncTimeoutSeconds `
            -Label 'global MTRand call synchronizer'

        if (-not (Test-Path -LiteralPath $globalMtrandCallSyncOutput)) {
            throw 'global MTRand call synchronizer result was not created'
        }
        $globalMtrandCallSyncResult = Get-Content `
            -Raw `
            -LiteralPath $globalMtrandCallSyncOutput |
            ConvertFrom-Json
        $handoffResumeCount = [int]$globalMtrandCallSyncResult.handoff_resume_previous_suspend_count
        if (
            $globalMtrandCallSyncResult.status -ne 'PASS' -or
            $globalMtrandCallSyncResult.stop_reason -ne
                'oracle_complete' -or
            [int]$globalMtrandCallSyncResult.expected_hit_count -le 0 -or
            [int]$globalMtrandCallSyncResult.hit_count -ne
                [int]$globalMtrandCallSyncResult.expected_hit_count -or
            [int]$globalMtrandCallSyncResult.missing_hit_count -ne 0 -or
            $null -ne $globalMtrandCallSyncResult.semantic_mismatch -or
            $null -ne (
                $globalMtrandCallSyncResult.unexpected_extra_call
            ) -or
            [int](
                $globalMtrandCallSyncResult.state_correction_count
            ) -ne 0 -or
            [int](
                $globalMtrandCallSyncResult.bytes_written_total
            ) -ne 0 -or
            -not [bool](
                $globalMtrandCallSyncResult.
                    final_post_verification.verified
            ) -or
            -not [bool](
                $globalMtrandCallSyncResult.hardware_breakpoint_restored
            ) -or
            -not [bool](
                $globalMtrandCallSyncResult.handoff_main_thread_resumed
            ) -or
            $handoffResumeCount -ne 1
        ) {
            throw (
                'global MTRand call synchronizer structured result ' +
                'did not pass'
            )
        }
    }

    Wait-OwnedProcess `
        -ProcessId $monitorProcess.Id `
        -TimeoutSeconds $MonitorTimeoutSeconds `
        -Label 'RNG monitor'

    $monitorLog = Join-Path $OutputRoot 'rng-monitor.log'
    if (-not (Test-Path -LiteralPath $monitorLog -PathType Leaf)) {
        throw 'RNG monitor result was not created'
    }
    $monitorResult = Get-Content -LiteralPath $monitorLog |
        Select-Object -Last 1 |
        ConvertFrom-Json
    if (
        $monitorResult.status -ne 'PASS' -or
        [int]$monitorResult.last_native_game_time -lt
            $StopAfterNativeGameTime
    ) {
        throw 'RNG monitor structured result did not pass'
    }
    if ($WaitForNaturalRuntimeExit) {
        Wait-OwnedProcess `
            -ProcessId $fullReplayMonitorProcess.Id `
            -TimeoutSeconds $RuntimeExitTimeoutSeconds `
            -Label 'full replay monitor'
        if (
            -not (
                Test-Path `
                    -LiteralPath $fullReplayMonitorOutput `
                    -PathType Leaf
            )
        ) {
            throw 'full replay monitor result was not created'
        }
        $fullReplayResult = Get-Content `
            -Raw `
            -LiteralPath $fullReplayMonitorOutput |
            ConvertFrom-Json
        $expectedFullReplaySnapshotCount = if ($HeadlessFullReplay) {
            0
        }
        else {
            6
        }
        $fullReplayOutcomeEvidencePass = if (
            $RequireSourceBoundGameplayParity
        ) {
            [bool](
                $fullReplayResult.natural_outcome_evidence_deferred
            ) -and
            $fullReplayResult.natural_outcome_evidence_mode -eq
                'source-bound-exact-gameplay-call-parity' -and
            $fullReplayResult.expected_outcome -eq $ExpectedOutcome -and
            (
                $ExpectedOutcome -ne 'natural_win' -or
                [int]$fullReplayResult.minimum_win_score -eq
                    $MinimumWinScore
            ) -and
            [int]$fullReplayResult.terminal_board_fallback_count -eq 0
        }
        else {
            $fullReplayResult.expected_outcome -eq $ExpectedOutcome -and
            $null -ne $fullReplayResult.natural_outcome -and
            (
                $ExpectedOutcome -ne 'natural_win' -or
                [int]$fullReplayResult.natural_outcome.score -ge
                    $MinimumWinScore
            ) -and
            (
                $ExpectedOutcome -ne 'natural_loss' -or
                [int]$fullReplayResult.natural_outcome.loss_counter -gt 0
            )
        }
        if (
            $fullReplayResult.status -ne 'PASS' -or
            [int]$fullReplayResult.exit_code -ne 0 -or
            -not [bool]$fullReplayResult.exit_observed -or
            [int]$fullReplayResult.snapshot_count -ne
                $expectedFullReplaySnapshotCount -or
            $null -eq $fullReplayResult.strict_trace_receipt -or
            [int]$fullReplayResult.command_order_offset -ne
                [int]$traceResult.result.command_order_offset -or
            [int]$fullReplayResult.process_memory_writes -ne 0 -or
            [bool]$fullReplayResult.persistent_file_modified -or
            -not $fullReplayOutcomeEvidencePass -or
            @(Get-ExactRuntimeProcesses).Count -ne 0
        ) {
            throw 'full replay structured result did not pass'
        }
        if ($RequireSourceBoundGameplayParity) {
            $sourceBoundParityOutput = Join-Path (
                $OutputRoot
            ) 'source-bound-gameplay-parity.json'
            & $Python $sourceBoundParityTool `
                --source-trace $GlobalCallRestoreTrace `
                --source-recording-report (
                    $GlobalCallRestoreRecordingReport
                ) `
                --replay-trace $hardwareGlobalCallTraceOutput `
                --strict-trace-result $resultPath `
                --full-replay-result $fullReplayMonitorOutput `
                --source-start-order "$GlobalCallRestoreSourceOrder" `
                --source-framework-update (
                    "$GlobalCallRestoreFrameworkUpdate"
                ) `
                --source-caller (
                    '0x{0:x8}' -f $GlobalCallRestoreCaller
                ) `
                --output $sourceBoundParityOutput
            if (-not (Test-Path -LiteralPath $sourceBoundParityOutput)) {
                throw 'source-bound gameplay parity result was not created'
            }
            $sourceBoundParityResult = Get-Content `
                -Raw `
                -LiteralPath $sourceBoundParityOutput |
                ConvertFrom-Json
            if (
                $sourceBoundParityResult.status -ne 'PASS' -or
                [int]$sourceBoundParityResult.exact_compared_call_count -le 0 -or
                $null -ne $sourceBoundParityResult.first_mismatch -or
                $sourceBoundParityResult.source.outcome -ne
                    $ExpectedOutcome -or
                $sourceBoundParityResult.replay.expected_outcome -ne
                    $ExpectedOutcome -or
                $null -eq $sourceBoundParityResult.replay.natural_outcome -or
                [bool]$sourceBoundParityResult.persistent_file_modified
            ) {
                throw 'source-bound gameplay parity result did not pass'
            }
        }
    }
}
catch {
    $runError = $_.Exception
}
finally {
    try {
        foreach (
            $process in @(
                $globalMtrandCallSyncProcess,
                $fullReplayMonitorProcess,
                $gameplayMtrandSyncProcess,
                $frameMtrandSyncProcess,
                $hardwareGlobalCallTraceProcess,
                $globalCallTraceProcess,
                $monitorProcess,
                $traceProcess
            )
        ) {
            if (
                $null -ne $process -and
                (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)
            ) {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }

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

            $hostRestoredPath = Join-Path $OutputRoot 'host-restored.json'
            & $Python $stateTool snapshot `
                --session-nonce $nonce `
                --phase host-restored `
                --output $hostRestoredPath | Out-Null

            $hostRoot = (
                Get-Content -Raw -LiteralPath $hostPrePath | ConvertFrom-Json
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
    expected_outcome = $ExpectedOutcome
    trace_result = Join-Path $OutputRoot 'result.json'
    rng_log = Join-Path $OutputRoot 'rng.ndjson'
    global_rng_call_trace = if ($globalCallTraceRequested) {
        Join-Path $OutputRoot 'global-rng-calls.json'
    } else {
        $null
    }
    hardware_global_rng_call_trace = if (
        $hardwareGlobalCallTraceRequested
    ) {
        Join-Path $OutputRoot 'global-mtrand-hardware-calls.json'
    } else {
        $null
    }
    frame_mtrand_sync = if ($frameMtrandSyncRequested) {
        Join-Path $OutputRoot 'frame-mtrand-sync.json'
    } else {
        $null
    }
    gameplay_mtrand_sync = if ($gameplayMtrandSyncRequested) {
        Join-Path $OutputRoot 'gameplay-mtrand-sync.json'
    } else {
        $null
    }
    global_mtrand_call_sync = if ($globalMtrandCallSyncRequested) {
        Join-Path $OutputRoot 'global-mtrand-call-sync.json'
    } else {
        $null
    }
    full_replay = if ($WaitForNaturalRuntimeExit) {
        Join-Path $OutputRoot 'full-replay.json'
    } else {
        $null
    }
    source_bound_gameplay_parity = if (
        $RequireSourceBoundGameplayParity
    ) {
        Join-Path $OutputRoot 'source-bound-gameplay-parity.json'
    } else {
        $null
    }
    host_root = if (Test-Path -LiteralPath $hostPrePath) {
        (
            Get-Content -Raw -LiteralPath $hostPrePath | ConvertFrom-Json
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
