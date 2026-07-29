param([string]$Profile = "fast", [int]$Seed = 42)
Write-Host "ActCIM-Robust Full Pipeline (Profile=$Profile, Seed=$Seed)"
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
$checkpoint = "results/baseline/seed_$Seed/best.pt"

Write-Host "1. Environment check..."
&amp; $python -m actcim_robust.cli check-env

Write-Host "2. Unit tests..."
&amp; $python -m pytest tests/ -v --tb=short

Write-Host "3. Baseline training..."
&amp; $python -m actcim_robust.cli train --config configs/baseline_fast.yaml --seed $Seed --profile $Profile --method baseline

Write-Host "4. Alpha sweep..."
&amp; $python -m actcim_robust.cli alpha-sweep --config configs/alpha_sweep.yaml --checkpoint $checkpoint

Write-Host "5. Layer sensitivity..."
&amp; $python -m actcim_robust.cli layer-sensitivity --config configs/layer_sensitivity.yaml --checkpoint $checkpoint

Write-Host "6. Random-NAT..."
&amp; $python -m actcim_robust.cli train --config configs/random_nat.yaml --seed $Seed --profile $Profile --method random_nat --checkpoint $checkpoint

Write-Host "7. SGR-NAT..."
&amp; $python -m actcim_robust.cli train --config configs/sgr_nat.yaml --seed $Seed --profile $Profile --method sgr_nat --checkpoint $checkpoint

Write-Host "Pipeline complete!"
