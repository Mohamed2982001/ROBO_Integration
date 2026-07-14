# Starts Qdrant if port 6333 is free, then runs the orchestrator from the project root.
# Usage (from repo root):  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_robot.ps1

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location $Root
$env:PYTHONUNBUFFERED = "1"

function Test-PortOpen([int] $Port) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", $Port)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

function Wait-PortOpen([int] $Port, [int] $TimeoutSec = 45) {
    $deadline = [datetime]::UtcNow.AddSeconds($TimeoutSec)
    while ([datetime]::UtcNow -lt $deadline) {
        if (Test-PortOpen -Port $Port) { return $true }
        Start-Sleep -Milliseconds 400
    }
    return $false
}

if (-not (Test-PortOpen -Port 6333)) {
    $qdrant = Join-Path $Root "qdrant.exe"
    if (-not (Test-Path $qdrant)) {
        Write-Host "[run] ERROR: port 6333 is closed and qdrant.exe was not found at $qdrant" -ForegroundColor Red
        exit 1
    }
    Write-Host "[run] Starting Qdrant (storage: .\storage\)..."
    Start-Process -FilePath $qdrant -WorkingDirectory $Root -WindowStyle Hidden
    if (-not (Wait-PortOpen -Port 6333)) {
        Write-Host "[run] ERROR: Qdrant did not open port 6333 in time." -ForegroundColor Red
        exit 1
    }
    Write-Host "[run] Qdrant is listening on 6333." -ForegroundColor Green
} else {
    Write-Host "[run] Qdrant already on 6333 -- skipping start." -ForegroundColor DarkGray
}

if (-not (Test-PortOpen -Port 27017)) {
    Write-Host "[run] WARN: mongod does not appear to be listening on 27017. Set MONGO_URI in .env or start MongoDB." -ForegroundColor Yellow
}

$py = "X:\Work\MiniConda\envs\tts\python.exe"
if (-not (Test-Path $py)) {
    $py = Join-Path $Root "venv311\Scripts\python.exe"
}
if (-not (Test-Path $py)) {
    $py = Join-Path $Root "venv\Scripts\python.exe"
}
if (-not (Test-Path $py)) {
    $py = "python"
}

$main = Join-Path $Root "main.py"
if (-not (Test-Path $main)) {
    Write-Host "[run] ERROR: main.py not found at $main" -ForegroundColor Red
    exit 1
}

Write-Host ('[run] Starting orchestrator: {0} -u main.py' -f $py) -ForegroundColor Cyan
& $py -u $main
exit $LASTEXITCODE
