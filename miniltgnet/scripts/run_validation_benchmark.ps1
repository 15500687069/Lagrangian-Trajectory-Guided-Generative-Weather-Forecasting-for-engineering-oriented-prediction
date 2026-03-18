param(
    [string]$Config = "miniltgnet/configs/lite_plus.yaml",
    [string]$Checkpoint = "outputs/miniltg/lite_plus/best.pt",
    [string]$Split = "test",
    [string]$OutputDir = "outputs/miniltg/validation",
    [string]$Methods = "model persistence linear climatology",
    [int]$MaxBatches = 0,
    [int]$ClimatologyMaxBatches = 0
)

$argsList = @(
    "miniltgnet/scripts/run_validation_benchmark.py",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--split", $Split,
    "--output_dir", $OutputDir,
    "--max_batches", "$MaxBatches",
    "--climatology_max_batches", "$ClimatologyMaxBatches",
    "--methods"
)

$argsList += $Methods.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)

python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Validation benchmark failed."
}

