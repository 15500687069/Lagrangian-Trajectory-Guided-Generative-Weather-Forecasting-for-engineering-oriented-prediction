$ErrorActionPreference = "Stop"

python miniltgnet/scripts/run_strict_benchmark.py `
  --config miniltgnet/configs/lite_plus_spec_finetune.yaml `
  --checkpoint outputs/miniltg/lite_plus_spec_finetune/selected/best_overall.pt `
  --split test `
  --methods model persistence linear climatology `
  --leads 1 2 3 4 5 6 `
  --bootstrap 2000 `
  --ci_alpha 0.05 `
  --reference persistence `
  --strict_p_threshold 0.95 `
  --output_dir outputs/miniltg/strict_benchmark
