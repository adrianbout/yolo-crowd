# SmartChairCounter - Docker GPU Setup Script for Windows
# This script automates the setup of Docker with GPU support on Windows
# Run this script as Administrator in PowerShell

# Requires: Windows 10/11, PowerShell 5.1+, Admin privileges

param(
    [switch]$SkipDriverCheck,
    [switch]$SkipDockerInstall,
    [switch]$SkipWSL
)

# Color functions
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-ColorOutput Green $args }
function Write-Info { Write-ColorOutput Cyan $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }

# Check if running as Administrator
function Test-Administrator {
    $currentUser = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
    return $currentUser.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    Write-Error "This script must be run as Administrator!"
    Write-Info "Right-click PowerShell and select 'Run as Administrator'"
    exit 1
}

Write-Info "========================================"
Write-Info "SmartChairCounter Docker GPU Setup"
Write-Info "========================================"
Write-Info ""

# Step 1: Check NVIDIA GPU
Write-Info "[1/5] Checking NVIDIA GPU..."
if (-not $SkipDriverCheck) {
    try {
        $nvidiaCheck = nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "[OK] NVIDIA GPU detected"
            Write-Info $nvidiaCheck | Select-Object -First 3
        } else {
            throw "nvidia-smi not found"
        }
    } catch {
        Write-Warning "[WARNING] NVIDIA driver not found or outdated"
        Write-Info "Please install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx"
        $response = Read-Host "Do you want to open the NVIDIA driver download page? (Y/N)"
        if ($response -eq 'Y' -or $response -eq 'y') {
            Start-Process "https://www.nvidia.com/Download/index.aspx"
        }
        Write-Error "Please install NVIDIA drivers (version 472.12+) and run this script again"
        exit 1
    }
} else {
    Write-Warning "Skipping driver check..."
}

# Step 2: Install/Check WSL 2
Write-Info ""
Write-Info "[2/5] Checking WSL 2..."
if (-not $SkipWSL) {
    try {
        $wslStatus = wsl --status 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "[OK] WSL 2 is installed"
        } else {
            throw "WSL not found"
        }
    } catch {
        Write-Warning "[WARNING] WSL 2 not found. Installing..."
        try {
            wsl --install
            Write-Success "[OK] WSL 2 installed successfully"
            Write-Warning "SYSTEM RESTART REQUIRED!"
            Write-Info "After restart, run this script again."
            $response = Read-Host "Restart now? (Y/N)"
            if ($response -eq 'Y' -or $response -eq 'y') {
                Restart-Computer -Force
            }
            exit 0
        } catch {
            Write-Error "Failed to install WSL 2: $_"
            exit 1
        }
    }

    # Set WSL 2 as default
    wsl --set-default-version 2 2>&1 | Out-Null
    Write-Success "[OK] WSL 2 set as default"
} else {
    Write-Warning "Skipping WSL check..."
}

# Step 3: Check/Install Docker Desktop
Write-Info ""
Write-Info "[3/5] Checking Docker Desktop..."
if (-not $SkipDockerInstall) {
    $dockerPath = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $dockerPath) {
        Write-Success "[OK] Docker Desktop is installed"
    } else {
        Write-Warning "[WARNING] Docker Desktop not found"
        Write-Info "Downloading Docker Desktop installer..."

        $installerPath = "$env:TEMP\DockerDesktopInstaller.exe"
        $downloadUrl = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"

        try {
            Write-Info "Downloading from: $downloadUrl"
            Invoke-WebRequest -Uri $downloadUrl -OutFile $installerPath -UseBasicParsing
            Write-Success "[OK] Download complete"

            Write-Info "Installing Docker Desktop..."
            Write-Warning "This may take several minutes..."
            Start-Process -FilePath $installerPath -ArgumentList "install", "--quiet" -Wait

            Write-Success "[OK] Docker Desktop installed"
            Write-Warning "SYSTEM RESTART REQUIRED!"
            Write-Info "After restart, run this script again."

            $response = Read-Host "Restart now? (Y/N)"
            if ($response -eq 'Y' -or $response -eq 'y') {
                Restart-Computer -Force
            }
            exit 0
        } catch {
            Write-Error "Failed to install Docker Desktop: $_"
            Write-Info "Please download and install manually from: https://www.docker.com/products/docker-desktop"
            exit 1
        }
    }

    # Check if Docker is running
    Write-Info "Checking if Docker is running..."
    try {
        docker version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Success "[OK] Docker is running"
        } else {
            throw "Docker not running"
        }
    } catch {
        Write-Warning "[WARNING] Docker Desktop is not running"
        Write-Info "Starting Docker Desktop..."
        Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        Write-Info "Waiting for Docker to start (this may take 30-60 seconds)..."

        $maxAttempts = 30
        $attempt = 0
        $dockerRunning = $false

        while ($attempt -lt $maxAttempts -and -not $dockerRunning) {
            Start-Sleep -Seconds 2
            $attempt++
            Write-Host "." -NoNewline
            $result = docker version 2>&1
            if ($LASTEXITCODE -eq 0) {
                $dockerRunning = $true
            }
        }

        if ($dockerRunning) {
            Write-Success "`n[OK] Docker started successfully"
        } else {
            Write-Error "`n[FAILED] Docker failed to start"
            Write-Info "Please start Docker Desktop manually and run this script again"
            exit 1
        }
    }
} else {
    Write-Warning "Skipping Docker installation check..."
}

# Step 4: Verify GPU support
Write-Info ""
Write-Info "[4/5] Verifying GPU support in Docker..."
Write-Info "Testing NVIDIA runtime (Docker Desktop handles GPU passthrough natively)..."

$maxRetries = 3
$retry = 0
$testPassed = $false

while ($retry -lt $maxRetries -and -not $testPassed) {
    try {
        Write-Info "Attempt $($retry + 1)/$maxRetries..."
        $gpuTest = docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            $testPassed = $true
            Write-Success "[OK] GPU support verified! Docker can access your NVIDIA GPU"
        } else {
            throw "Test failed: $gpuTest"
        }
    } catch {
        $retry++
        if ($retry -lt $maxRetries) {
            Write-Warning "Test failed, retrying in 5 seconds..."
            Start-Sleep -Seconds 5
        }
    }
}

if (-not $testPassed) {
    Write-Error "[FAILED] GPU test failed"
    Write-Warning "Docker cannot access the GPU. Please check:"
    Write-Info "  1. Docker Desktop is running with WSL 2 backend enabled"
    Write-Info "  2. In Docker Desktop Settings > Resources > WSL Integration - enable your distro"
    Write-Info "  3. NVIDIA drivers are up to date (472.12+)"
    Write-Info "  4. Try restarting Docker Desktop and your computer"
    exit 1
} else {
    # Show GPU info
    Write-Info ""
    Write-Info "GPU Information:"
    docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
}

# Step 5: Build the SmartChairCounter image
Write-Info ""
Write-Info "[5/5] Building SmartChairCounter Docker image..."
$projectPath = $PSScriptRoot

if (Test-Path "$projectPath\docker-compose.yml") {
    Write-Info "Found docker-compose.yml, building with GPU support..."

    Set-Location $projectPath

    Write-Info "This may take several minutes on first build..."
    docker compose build

    if ($LASTEXITCODE -eq 0) {
        Write-Success "[OK] Image built successfully!"
        Write-Info ""
        Write-Info "========================================"
        Write-Success "Setup Complete!"
        Write-Info "========================================"
        Write-Info ""
        Write-Info "To start the application, run:"
        Write-Success "  docker compose up -d"
        Write-Info ""
        Write-Info "To view logs:"
        Write-Success "  docker compose logs -f"
        Write-Info ""
        Write-Info "To check GPU usage:"
        Write-Success "  docker exec smartchair-counter nvidia-smi"
        Write-Info ""
        Write-Info "Access the application at:"
        Write-Success "  http://localhost:8000"
        Write-Info ""

        $response = Read-Host "Do you want to start the application now? (Y/N)"
        if ($response -eq 'Y' -or $response -eq 'y') {
            docker compose up -d
            Start-Sleep -Seconds 5
            Write-Success "Application started!"
            Write-Info "Opening browser..."
            Start-Process "http://localhost:8000"
        }
    } else {
        Write-Error "Build failed. Please check the error messages above."
        exit 1
    }
} else {
    Write-Warning "docker-compose.yml not found in current directory"
    Write-Info "Please run this script from the project root directory"
    exit 1
}

Write-Info ""
Write-Success "All done!"
