param([string]$Checkpoint = "results/baseline/seed_42/best.pt")
Write-Host "Running alpha sweep..."
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
&amp; $python -m actcim_robust.cli alpha-sweep --config configs/alpha_sweep.yaml --checkpoint $Checkpoint
