# SmartChairCounter - Docker Management Script
# Simple script to manage your Docker container

param(
    [Parameter(Position=0)]
    [ValidateSet('start', 'stop', 'restart', 'logs', 'status', 'build', 'rebuild', 'shell', 'gpu', 'clean', 'help')]
    [string]$Command = 'help'
)

$ProjectName = "smartchair-counter"
$ComposeFile = "docker-compose.yml"

function Write-Title {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host " SmartChairCounter - Docker Manager" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Show-Help {
    Write-Title
    Write-Host "Usage: .\docker-manage.ps1 <command>" -ForegroundColor Yellow
    Write-Host "`nAvailable commands:" -ForegroundColor Green
    Write-Host "  start    " -ForegroundColor Yellow -NoNewline
    Write-Host "  - Start the application"
    Write-Host "  stop     " -ForegroundColor Yellow -NoNewline
    Write-Host "  - Stop the application"
    Write-Host "  restart  " -ForegroundColor Yellow -NoNewline
    Write-Host "  - Restart the application"
    Write-Host "  logs     " -ForegroundColor Yellow -NoNewline
    Write-Host "  - View application logs"
    Write-Host "  status   " -ForegroundColor Yellow -NoNewline
    Write-Host "  - Check application status"
    Write-Host "  build    " -ForegroundColor Yellow -NoNewline
    Write-Host "  - Build the Docker image"
    Write-Host "  rebuild  " -ForegroundColor Yellow -NoNewline
    Write-Host "  - Rebuild from scratch (no cache)"
    Write-Host "  shell    " -ForegroundColor Yellow -NoNewline
    Write-Host "  - Open shell in container"
    Write-Host "  gpu      " -ForegroundColor Yellow -NoNewline
    Write-Host "  - Check GPU status"
    Write-Host "  clean    " -ForegroundColor Yellow -NoNewline
    Write-Host "  - Remove all containers and images"
    Write-Host "  help     " -ForegroundColor Yellow -NoNewline
    Write-Host "  - Show this help message"
    Write-Host "`nExamples:" -ForegroundColor Green
    Write-Host "  .\docker-manage.ps1 start" -ForegroundColor Cyan
    Write-Host "  .\docker-manage.ps1 logs" -ForegroundColor Cyan
    Write-Host "  .\docker-manage.ps1 gpu`n" -ForegroundColor Cyan
}

function Start-Application {
    Write-Title
    Write-Host "Starting SmartChairCounter..." -ForegroundColor Green
    docker-compose up -d

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Application started successfully!" -ForegroundColor Green
        Write-Host "`nAccess the application at:" -ForegroundColor Cyan
        Write-Host "  • API: " -ForegroundColor Yellow -NoNewline
        Write-Host "http://localhost:8000" -ForegroundColor White
        Write-Host "  • Health: " -ForegroundColor Yellow -NoNewline
        Write-Host "http://localhost:8000/health" -ForegroundColor White
        Write-Host "  • Docs: " -ForegroundColor Yellow -NoNewline
        Write-Host "http://localhost:8000/docs" -ForegroundColor White
        Write-Host "  • Frontend: " -ForegroundColor Yellow -NoNewline
        Write-Host "http://localhost:8000/frontend/`n" -ForegroundColor White

        $response = Read-Host "Open in browser? (Y/N)"
        if ($response -eq 'Y' -or $response -eq 'y') {
            Start-Process "http://localhost:8000"
        }
    } else {
        Write-Host "✗ Failed to start application" -ForegroundColor Red
        Write-Host "Run '.\docker-manage.ps1 logs' to see errors`n" -ForegroundColor Yellow
    }
}

function Stop-Application {
    Write-Title
    Write-Host "Stopping SmartChairCounter..." -ForegroundColor Yellow
    docker-compose down

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Application stopped`n" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to stop application`n" -ForegroundColor Red
    }
}

function Restart-Application {
    Write-Title
    Write-Host "Restarting SmartChairCounter..." -ForegroundColor Yellow
    docker-compose restart

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Application restarted`n" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to restart application`n" -ForegroundColor Red
    }
}

function Show-Logs {
    Write-Title
    Write-Host "Showing logs (Ctrl+C to exit)..." -ForegroundColor Cyan
    Write-Host ""
    docker-compose logs -f
}

function Show-Status {
    Write-Title
    Write-Host "Container Status:" -ForegroundColor Green
    docker-compose ps

    Write-Host "`nDocker Stats:" -ForegroundColor Green
    docker stats --no-stream $ProjectName 2>$null

    Write-Host "`nHealth Check:" -ForegroundColor Green
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        $health = $response.Content | ConvertFrom-Json
        Write-Host "Status: " -ForegroundColor Yellow -NoNewline
        Write-Host $health.status -ForegroundColor Green
        Write-Host "Detection Service: " -ForegroundColor Yellow -NoNewline
        Write-Host $health.detection_service -ForegroundColor $(if($health.detection_service) {"Green"} else {"Red"})
    } catch {
        Write-Host "✗ Application not responding" -ForegroundColor Red
        Write-Host "  Container may be stopped or still starting`n" -ForegroundColor Yellow
    }
}

function Build-Image {
    Write-Title
    Write-Host "Building Docker image..." -ForegroundColor Green
    Write-Host "This may take 10-20 minutes on first build...`n" -ForegroundColor Yellow
    docker-compose build

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Build completed successfully!" -ForegroundColor Green
        Write-Host "Run '.\docker-manage.ps1 start' to start the application`n" -ForegroundColor Cyan
    } else {
        Write-Host "`n✗ Build failed`n" -ForegroundColor Red
    }
}

function Rebuild-Image {
    Write-Title
    Write-Host "Rebuilding Docker image from scratch..." -ForegroundColor Yellow
    Write-Host "This will take longer as it rebuilds everything...`n" -ForegroundColor Yellow
    docker-compose build --no-cache

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Rebuild completed successfully!" -ForegroundColor Green
        Write-Host "Run '.\docker-manage.ps1 start' to start the application`n" -ForegroundColor Cyan
    } else {
        Write-Host "`n✗ Rebuild failed`n" -ForegroundColor Red
    }
}

function Open-Shell {
    Write-Title
    Write-Host "Opening shell in container..." -ForegroundColor Green
    Write-Host "Type 'exit' to return to Windows`n" -ForegroundColor Yellow
    docker exec -it $ProjectName bash

    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n✗ Failed to open shell" -ForegroundColor Red
        Write-Host "Container may not be running. Run '.\docker-manage.ps1 start' first`n" -ForegroundColor Yellow
    }
}

function Check-GPU {
    Write-Title
    Write-Host "GPU Status:`n" -ForegroundColor Green

    # Check NVIDIA driver on host
    Write-Host "Host GPU:" -ForegroundColor Cyan
    try {
        nvidia-smi
        Write-Host ""
    } catch {
        Write-Host "✗ NVIDIA driver not found on host" -ForegroundColor Red
        Write-Host "Install from: https://www.nvidia.com/Download/index.aspx`n" -ForegroundColor Yellow
        return
    }

    # Check GPU in container
    Write-Host "`nContainer GPU Access:" -ForegroundColor Cyan
    docker exec $ProjectName nvidia-smi 2>$null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Container can access GPU" -ForegroundColor Green
    } else {
        Write-Host "✗ Container cannot access GPU" -ForegroundColor Red
        Write-Host "Run the setup script: .\setup-docker-gpu.ps1`n" -ForegroundColor Yellow
        return
    }

    # Check PyTorch CUDA
    Write-Host "`nPyTorch CUDA Status:" -ForegroundColor Cyan
    docker exec $ProjectName python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device Count: {torch.cuda.device_count()}'); print(f'Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>$null

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ PyTorch can use GPU`n" -ForegroundColor Green
    } else {
        Write-Host "`n✗ PyTorch cannot access GPU" -ForegroundColor Red
        Write-Host "Container may not be running. Run '.\docker-manage.ps1 start' first`n" -ForegroundColor Yellow
    }
}

function Clean-All {
    Write-Title
    Write-Host "This will remove:" -ForegroundColor Yellow
    Write-Host "  • All SmartChairCounter containers" -ForegroundColor Red
    Write-Host "  • All SmartChairCounter images" -ForegroundColor Red
    Write-Host "  • All unused Docker volumes" -ForegroundColor Red
    Write-Host "`nYour config, weights, and source code will NOT be deleted.`n" -ForegroundColor Green

    $response = Read-Host "Are you sure? (yes/no)"
    if ($response -eq 'yes') {
        Write-Host "`nCleaning up..." -ForegroundColor Yellow

        # Stop and remove containers
        docker-compose down -v

        # Remove images
        docker rmi $(docker images | Where-Object {$_ -match 'smartchair'} | ForEach-Object {($_ -split '\s+')[2]}) 2>$null

        # Prune unused volumes
        docker volume prune -f

        Write-Host "✓ Cleanup complete`n" -ForegroundColor Green
    } else {
        Write-Host "Cleanup cancelled`n" -ForegroundColor Yellow
    }
}

# Main script logic
switch ($Command) {
    'start'   { Start-Application }
    'stop'    { Stop-Application }
    'restart' { Restart-Application }
    'logs'    { Show-Logs }
    'status'  { Show-Status }
    'build'   { Build-Image }
    'rebuild' { Rebuild-Image }
    'shell'   { Open-Shell }
    'gpu'     { Check-GPU }
    'clean'   { Clean-All }
    'help'    { Show-Help }
    default   { Show-Help }
}
