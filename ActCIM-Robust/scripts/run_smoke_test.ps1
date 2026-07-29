Write-Host "Running smoke test..."
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
&amp; $python -m actcim_robust.cli train --config configs/smoke.yaml --seed 42 --profile smoke --method baseline
