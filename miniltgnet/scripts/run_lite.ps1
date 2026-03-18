[CmdletBinding()]
param(
    [string]$Config = "miniltgnet/configs/lite.yaml",
    [string]$SanityConfig = "miniltgnet/configs/lite_sanity.yaml",
    [string]$EvalSplit = "test",
    [switch]$SkipPrepare,
    [switch]$SkipSanity
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked([string]$Exe, [string[]]$Args) {
    Write-Host ""
    Write-Host ">>> $Exe $($Args -join ' ')" -ForegroundColor DarkGray
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $Exe $($Args -join ' ')"
    }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
Set-Location $repoRoot

Write-Host "Repository: $repoRoot"
Invoke-Checked "python" @("--version")

if (-not $SkipPrepare) {
    Invoke-Checked "python" @("-m", "miniltgnet.cli", "prepare", "--config", $Config)
}

if (-not $SkipSanity) {
    Invoke-Checked "python" @("-m", "miniltgnet.cli", "sanity", "--config", $SanityConfig, "--steps", "2", "--backward")
}

Invoke-Checked "python" @("-m", "miniltgnet.cli", "train", "--config", $Config)
Invoke-Checked "python" @("-m", "miniltgnet.cli", "evaluate", "--config", $Config, "--checkpoint", "outputs/miniltg/lite/best.pt", "--split", $EvalSplit)

Write-Host ""
Write-Host "Done. Output: outputs/miniltg/lite/"
