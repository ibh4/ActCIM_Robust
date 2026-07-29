param([string]$Checkpoint = "results/baseline/seed_42/best.pt")
Write-Host "Running layer sensitivity analysis..."
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
&amp; $python -m actcim_robust.cli layer-sensitivity --config configs/layer_sensitivity.yaml --checkpoint $Checkpoint
