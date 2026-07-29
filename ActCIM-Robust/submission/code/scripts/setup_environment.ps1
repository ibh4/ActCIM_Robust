# Setup environment script
Write-Host "Setting up ActCIM-Robust environment..."
$python = "C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"

Write-Host "Checking Python..."
&amp; $python --version

Write-Host "Running environment check..."
&amp; $python -m actcim_robust.cli check-env

Write-Host "Setup complete!"
