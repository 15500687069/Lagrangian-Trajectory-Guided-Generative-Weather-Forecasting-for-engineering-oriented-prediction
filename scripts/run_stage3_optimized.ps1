[CmdletBinding()]
param(
    [string]$PrepareConfig = "configs/stage1_regional.yaml",
    [string]$RecoverConfig = "configs/stage3_recover.yaml",
    [string]$FinetuneConfig = "configs/stage3_finetune.yaml",
    [string]$SanityConfig = "configs/stage3_redesign_sanity.yaml",
    [string]$EvalSplit = "test",
    [switch]$SkipPrepare,
    [switch]$SkipSanity,
    [int]$SanitySteps = 2
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

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Repository: $repoRoot"
Invoke-Checked "python" @("--version")

if (-not $SkipPrepare) {
    Invoke-Checked "python" @("-m", "ltg_net.cli", "prepare", "--config", $PrepareConfig)
}

if (-not $SkipSanity) {
    Invoke-Checked "python" @("scripts/sanity_check.py", "--config", $SanityConfig, "--steps", "$SanitySteps", "--backward")
}

Invoke-Checked "python" @("-m", "ltg_net.cli", "train", "--config", $RecoverConfig)
Invoke-Checked "python" @("-m", "ltg_net.cli", "evaluate", "--config", $RecoverConfig, "--checkpoint", "outputs/stage3_recover/best.pt", "--split", $EvalSplit)

Invoke-Checked "python" @("-m", "ltg_net.cli", "train", "--config", $FinetuneConfig)
Invoke-Checked "python" @("-m", "ltg_net.cli", "evaluate", "--config", $FinetuneConfig, "--checkpoint", "outputs/stage3_finetune/best.pt", "--split", $EvalSplit)

Invoke-Checked "python" @(
    "scripts/export_metrics_summary.py",
    "--inputs", "outputs/stage3_recover/test_metrics.jsonl", "outputs/stage3_finetune/test_metrics.jsonl",
    "--output", "outputs/stage3_metrics_summary.csv"
)

Write-Host ""
Write-Host "Done. Key outputs:"
Write-Host "  outputs/stage3_recover/"
Write-Host "  outputs/stage3_finetune/"
Write-Host "  outputs/stage3_metrics_summary.csv"
