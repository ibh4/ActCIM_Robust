param([int]$Seed = 42)
Write-Host "Running baseline training (seed=$Seed)..."
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
&amp; $python -m actcim_robust.cli train --config configs/baseline_fast.yaml --seed $Seed --profile fast --method baseline
