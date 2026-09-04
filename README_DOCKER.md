# SmartChairCounter - Docker Deployment with GPU Support

Complete Docker setup for running SmartChairCounter on Windows with NVIDIA GPU acceleration.

## 📋 What's Included

This Docker setup includes:
- ✅ NVIDIA GPU Support (CUDA 12.1)
- ✅ All Python dependencies (PyTorch, Ultralytics, FastAPI, etc.)
- ✅ Automated setup scripts
- ✅ Management utilities
- ✅ Volume persistence for config, weights, and logs

## 🚀 Quick Start (3 Steps)

### Step 1: Prerequisites
- Windows 10/11 (21H2 or newer)
- NVIDIA GPU (GTX 10-series or newer)
- 20GB free disk space

### Step 2: Run Setup Script
Open **PowerShell as Administrator**:

```powershell
cd "d:\Paradox\Chair Project\SmartChairCounter\yolo-crowd"
.\setup-docker-gpu.ps1
```

This automatically installs everything needed (WSL 2, Docker, NVIDIA Container Toolkit, etc.)

### Step 3: Done!
Access your application at: **http://localhost:8000**

---

## 📁 Files Created

| File | Purpose |
|------|---------|
| `Dockerfile` | CPU-only Docker image (basic) |
| `Dockerfile.gpu` | GPU-enabled Docker image (recommended) |
| `.dockerignore` | Excludes unnecessary files from image |
| `docker-compose.yml` | Docker Compose configuration (GPU-enabled) |
| `setup-docker-gpu.ps1` | **Automated setup script** |
| `docker-manage.ps1` | Container management utilities |
| `QUICK_START.md` | Quick start guide |
| `DOCKER_INSTRUCTIONS.md` | Detailed Docker instructions |
| `README_DOCKER.md` | This file |

---

## 🎮 Management Commands

Use the management script for easy control:

```powershell
# Start the application
.\docker-manage.ps1 start

# Stop the application
.\docker-manage.ps1 stop

# View logs (live)
.\docker-manage.ps1 logs

# Check status and health
.\docker-manage.ps1 status

# Verify GPU is working
.\docker-manage.ps1 gpu

# Open shell inside container
.\docker-manage.ps1 shell

# Rebuild image
.\docker-manage.ps1 rebuild

# Clean up everything
.\docker-manage.ps1 clean
```

Or use Docker Compose directly:

```powershell
docker-compose up -d      # Start
docker-compose down       # Stop
docker-compose logs -f    # View logs
docker-compose ps         # Check status
```

---

## 🔧 Manual Setup (Alternative)

If you prefer manual setup or the script doesn't work:

### 1. Install Prerequisites

**NVIDIA Drivers (472.12+)**
- Download: https://www.nvidia.com/Download/index.aspx
- Verify: `nvidia-smi`

**WSL 2**
```powershell
wsl --install
wsl --set-default-version 2
```

**Docker Desktop**
- Download: https://www.docker.com/products/docker-desktop
- Enable WSL 2 backend in settings

### 2. Install NVIDIA Container Toolkit

Open WSL terminal:

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```

Restart Docker Desktop.

### 3. Verify GPU Support

```powershell
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

You should see your GPU info.

### 4. Build and Run

```powershell
cd "d:\Paradox\Chair Project\SmartChairCounter\yolo-crowd"
docker-compose up -d --build
```

---

## 🌐 Access Points

Once running, access the application:

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Health Check | http://localhost:8000/health |
| API Documentation | http://localhost:8000/docs |
| Frontend | http://localhost:8000/frontend/ |

---

## 📊 Monitoring

### Check GPU Usage (Live)
```powershell
# From Windows
nvidia-smi -l 1

# Inside container
docker exec smartchair-counter nvidia-smi
```

### Check Container Logs
```powershell
docker-compose logs -f
```

### Check if PyTorch Sees GPU
```powershell
docker exec smartchair-counter python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output: `CUDA: True`

---

## 🔍 Troubleshooting

### Container won't start
```powershell
# Check logs for errors
docker-compose logs

# Check if port 8000 is in use
netstat -ano | findstr :8000
```

### GPU not detected
```powershell
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Verify container GPU access
.\docker-manage.ps1 gpu
```

### Port 8000 already in use
Edit [docker-compose.yml](docker-compose.yml) line 10:
```yaml
ports:
  - "8080:8000"  # Change to any available port
```

### Out of memory errors
Reduce batch size in [backend/main.py](backend/main.py) line 36:
```python
detection_service = DetectionService(state_manager, batch_size=1, inference_interval=0.0)
```

### Build is very slow
- Check internet connection
- Try: `docker-compose build --no-cache`
- Increase Docker memory: Docker Desktop → Settings → Resources

### Script execution disabled
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📦 What Gets Mounted

Volumes persist these directories:

| Host Directory | Container Path | Purpose |
|----------------|----------------|---------|
| `./config` | `/app/config` | Camera & ROI configurations |
| `./weights` | `/app/weights` | YOLO model weights (.pt files) |
| `./frontend` | `/app/frontend` | Web interface files |
| `./logs` | `/app/logs` | Application logs |

Your data is safe even if you rebuild or remove containers.

---

## ⚡ Performance Tips

1. **Increase batch size** for better GPU utilization:
   ```python
   # In backend/main.py
   detection_service = DetectionService(state_manager, batch_size=4, ...)
   ```

2. **Monitor GPU memory**:
   ```powershell
   nvidia-smi -l 1
   ```

3. **Use SSD** for weights directory

4. **Allocate more RAM** to Docker:
   Docker Desktop → Settings → Resources → Memory

---

## 🗑️ Cleanup

Remove everything:
```powershell
# Using management script
.\docker-manage.ps1 clean

# Or manually
docker-compose down -v
docker rmi $(docker images -q smartchair*)
```

Your source code, config, and weights are NOT deleted.

---

## 📚 Additional Documentation

- [QUICK_START.md](QUICK_START.md) - Fast setup guide
- [DOCKER_INSTRUCTIONS.md](DOCKER_INSTRUCTIONS.md) - Detailed instructions
- [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md) - GPU setup details (if created)

---

## 🆘 Getting Help

If you encounter issues:

1. Check logs: `.\docker-manage.ps1 logs`
2. Check status: `.\docker-manage.ps1 status`
3. Verify GPU: `.\docker-manage.ps1 gpu`
4. Check Docker is running: `docker ps`
5. Verify NVIDIA driver: `nvidia-smi`

---

## 📝 Notes

- First build takes 10-20 minutes (downloads all dependencies)
- Subsequent builds are much faster (uses cache)
- GPU support requires NVIDIA GPU with CUDA 3.5+
- The setup uses CUDA 12.1 (compatible with PyTorch 2.9.1)
- WSL 2 provides near-native GPU performance

---

## 🎯 What the Setup Script Does

The `setup-docker-gpu.ps1` script automates:

1. ✅ Checks for NVIDIA GPU and drivers
2. ✅ Installs WSL 2 (if needed)
3. ✅ Installs Docker Desktop (if needed)
4. ✅ Installs NVIDIA Container Toolkit in WSL
5. ✅ Configures Docker for GPU support
6. ✅ Verifies GPU access in Docker
7. ✅ Builds the SmartChairCounter image
8. ✅ Optionally starts the application

Total setup time: 20-30 minutes (mostly downloads)

---

Made with ❤️ for SmartChairCounter
