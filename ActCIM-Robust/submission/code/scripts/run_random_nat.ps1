param([int]$Seed = 42, [string]$Checkpoint = "results/baseline/seed_42/best.pt")
Write-Host "Running Random-NAT (seed=$Seed)..."
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
&amp; $python -m actcim_robust.cli train --config configs/random_nat.yaml --seed $Seed --profile fast --method random_nat --checkpoint $Checkpoint
