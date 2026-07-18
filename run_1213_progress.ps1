$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Log = Join-Path $Root "run_1213_progress.log"

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "0"

Set-Location -LiteralPath $Root
if (Test-Path -LiteralPath $Log) {
    Remove-Item -LiteralPath $Log
}

Start-Transcript -Path $Log -Force | Out-Null

function Write-LogLine {
    param([string]$Message)
    Write-Host ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message)
}

function Run-Step {
    param(
        [string]$Name,
        [string]$ScriptPath
    )
    Write-LogLine "START $Name"
    & $Python $ScriptPath
    if ($LASTEXITCODE -ne 0) {
        Write-LogLine "FAILED $Name exit_code=$LASTEXITCODE"
        Stop-Transcript | Out-Null
        exit $LASTEXITCODE
    }
    Write-LogLine "DONE $Name"
}

Write-LogLine "RUN Chapter 12 and Chapter 13"
Write-LogLine "python=$Python"

Run-Step "Chapter 12 knock90" (Join-Path $Root "Chapter 12\knock90.py")
Run-Step "Chapter 12 knock91" (Join-Path $Root "Chapter 12\knock91.py")
Run-Step "Chapter 12 knock92" (Join-Path $Root "Chapter 12\knock92.py")
Run-Step "Chapter 12 knock93" (Join-Path $Root "Chapter 12\knock93.py")
Run-Step "Chapter 12 knock94" (Join-Path $Root "Chapter 12\knock94.py")
Run-Step "Chapter 12 knock95" (Join-Path $Root "Chapter 12\knock95.py")
Run-Step "Chapter 12 knock96" (Join-Path $Root "Chapter 12\knock96.py")
Run-Step "Chapter 12 knock97" (Join-Path $Root "Chapter 12\knock97.py")
Run-Step "Chapter 12 knock98" (Join-Path $Root "Chapter 12\knock98.py")
Run-Step "Chapter 12 knock99" (Join-Path $Root "Chapter 12\knock99.py")

Run-Step "Chapter 13 knock90" (Join-Path $Root "Chapter 13\knock90.py")
Run-Step "Chapter 13 knock91" (Join-Path $Root "Chapter 13\knock91.py")
Run-Step "Chapter 13 knock92" (Join-Path $Root "Chapter 13\knock92.py")
Run-Step "Chapter 13 knock93" (Join-Path $Root "Chapter 13\knock93.py")

Write-LogLine "ALL DONE"
Stop-Transcript | Out-Null
