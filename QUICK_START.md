# Quick Start Guide - Docker GPU Setup

## Automated Setup (Recommended)

### For Windows Users:

**Option 1: Run the automated setup script (installs everything)**

1. Open **PowerShell as Administrator** (Right-click PowerShell → Run as Administrator)

2. Navigate to the project directory:
   ```powershell
   cd "d:\Paradox\Chair Project\SmartChairCounter\yolo-crowd"
   ```

3. Run the setup script:
   ```powershell
   .\setup-docker-gpu.ps1
   ```

   This script will:
   - ✓ Check for NVIDIA GPU and drivers
   - ✓ Install WSL 2 (if needed)
   - ✓ Install Docker Desktop (if needed)
   - ✓ Install NVIDIA Container Toolkit in WSL
   - ✓ Configure Docker for GPU support
   - ✓ Build the SmartChairCounter image
   - ✓ Start the application

4. **That's it!** The application will be running at: http://localhost:8000

---

**Option 2: If Docker is already installed, quick build & run:**

```powershell
# Navigate to project directory
cd "d:\Paradox\Chair Project\SmartChairCounter\yolo-crowd"

# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f
```

---

## Manual Setup Steps

If the automated script doesn't work, follow these steps:

### 1. Install Prerequisites

- **NVIDIA Drivers (472.12+)**: https://www.nvidia.com/Download/index.aspx
- **Docker Desktop**: https://www.docker.com/products/docker-desktop
- **WSL 2**: Run in PowerShell (Admin): `wsl --install`

### 2. Install NVIDIA Container Toolkit

Open WSL terminal and run:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```

Restart Docker Desktop after installation.

### 3. Verify GPU Support

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

You should see your GPU information.

### 4. Build and Run

```powershell
cd "d:\Paradox\Chair Project\SmartChairCounter\yolo-crowd"
docker-compose up -d --build
```

---

## Verification

### Check if container is running:
```powershell
docker ps
```

### Check GPU is detected:
```powershell
docker exec smartchair-counter nvidia-smi
```

### Check PyTorch can use GPU:
```powershell
docker exec smartchair-counter python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output: `CUDA Available: True`

### View logs:
```powershell
docker-compose logs -f
```

---

## Access the Application

- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Documentation**: http://localhost:8000/docs
- **Frontend**: http://localhost:8000/frontend/

---

## Common Commands

```powershell
# Start the application
docker-compose up -d

# Stop the application
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Rebuild from scratch
docker-compose build --no-cache

# Check GPU usage
nvidia-smi -l 1

# Access container shell
docker exec -it smartchair-counter bash
```

---

## Troubleshooting

### Problem: Script execution is disabled

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: Docker can't access GPU

**Solutions:**
1. Verify NVIDIA drivers: `nvidia-smi`
2. Restart Docker Desktop
3. Restart computer
4. Check Docker uses WSL 2 backend (Docker Desktop → Settings → General)

### Problem: Port 8000 already in use

**Solution:** Change port in [docker-compose.yml](d:\Paradox\Chair Project\SmartChairCounter\yolo-crowd\docker-compose.yml):
```yaml
ports:
  - "8080:8000"  # Use port 8080 instead
```

### Problem: Build takes too long or fails

**Solution:**
- Check internet connection
- Try: `docker-compose build --no-cache`
- Increase Docker memory: Docker Desktop → Settings → Resources

---

## Getting Help

If you encounter issues:

1. Check logs: `docker-compose logs`
2. Check Docker is running: `docker ps`
3. Verify GPU access: `nvidia-smi`
4. Check WSL 2: `wsl --status`

For more detailed information, see:
- [GPU_SETUP_GUIDE.md](./GPU_SETUP_GUIDE.md) - Comprehensive GPU setup guide
- [DOCKER_INSTRUCTIONS.md](./DOCKER_INSTRUCTIONS.md) - Detailed Docker instructions

---

## Performance Tips

- Monitor GPU usage: `nvidia-smi -l 1`
- Increase batch size for better GPU utilization (edit `backend/main.py`)
- Use SSD for weights storage
- Allocate more RAM to Docker (Settings → Resources)
