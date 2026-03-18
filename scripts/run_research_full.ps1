[CmdletBinding()]
param(
    [string]$PrepareConfig = "configs/stage1_regional.yaml",
    [string]$Stage1Config = "configs/stage1_regional.yaml",
    [string]$Stage2Config = "configs/stage2_redesign.yaml",
    [string]$Stage3RecoverConfig = "configs/stage3_recover.yaml",
    [string]$Stage3FinetuneConfig = "configs/stage3_finetune.yaml",
    [string]$EvalSplit = "test",
    [switch]$SkipPrepare,
    [switch]$SkipStage1,
    [switch]$SkipStage2,
    [switch]$SkipStage3
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

function Run-Stage([string]$ConfigPath, [string]$CheckpointPath, [string]$Split) {
    Invoke-Checked "python" @("-m", "ltg_net.cli", "train", "--config", $ConfigPath)
    Invoke-Checked "python" @("-m", "ltg_net.cli", "evaluate", "--config", $ConfigPath, "--checkpoint", $CheckpointPath, "--split", $Split)
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "Repository: $repoRoot"
Invoke-Checked "python" @("--version")

if (-not $SkipPrepare) {
    Invoke-Checked "python" @("-m", "ltg_net.cli", "prepare", "--config", $PrepareConfig)
}

if (-not $SkipStage1) {
    Run-Stage -ConfigPath $Stage1Config -CheckpointPath "outputs/stage1/best.pt" -Split $EvalSplit
}

if (-not $SkipStage2) {
    Run-Stage -ConfigPath $Stage2Config -CheckpointPath "outputs/stage2_redesign/best.pt" -Split $EvalSplit
}

if (-not $SkipStage3) {
    Run-Stage -ConfigPath $Stage3RecoverConfig -CheckpointPath "outputs/stage3_recover/best.pt" -Split $EvalSplit
    Run-Stage -ConfigPath $Stage3FinetuneConfig -CheckpointPath "outputs/stage3_finetune/best.pt" -Split $EvalSplit
    Invoke-Checked "python" @(
        "scripts/export_metrics_summary.py",
        "--inputs", "outputs/stage3_recover/test_metrics.jsonl", "outputs/stage3_finetune/test_metrics.jsonl",
        "--output", "outputs/stage3_metrics_summary.csv"
    )
}

Write-Host ""
Write-Host "Done."
