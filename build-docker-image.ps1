# Build and export Docker image for distribution
# This creates a Docker image that can be loaded on any Docker Desktop

$version = "1.0.9"
$imageName = "smartchair-counter"
$imageTag = "${imageName}:${version}"
$exportFile = "smartchair-counter-${version}.tar"

Write-Host "Building Docker image: $imageTag" -ForegroundColor Green
Write-Host "This may take several minutes..." -ForegroundColor Yellow

# Build the Docker image (CPU version for compatibility)
docker build -t $imageTag -f Dockerfile .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nBuild successful!" -ForegroundColor Green
Write-Host "Exporting Docker image to tar file..." -ForegroundColor Yellow

# Save the Docker image to a tar file
docker save -o $exportFile $imageTag

if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker save failed!" -ForegroundColor Red
    exit 1
}

# Show file size
$fileSize = (Get-Item $exportFile).Length / 1MB
Write-Host "`nDocker image exported successfully!" -ForegroundColor Green
Write-Host "File: $exportFile" -ForegroundColor Cyan
Write-Host "Size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Cyan

Write-Host "`n=== To use this image on another machine ===" -ForegroundColor Yellow
Write-Host "1. Copy $exportFile to the target machine" -ForegroundColor White
Write-Host "2. Load the image: docker load -i $exportFile" -ForegroundColor White
Write-Host "3. Run the container: docker run -d -p 8000:8000 --name smartchair $imageTag" -ForegroundColor White
Write-Host "`nOr use docker-compose with the image already loaded." -ForegroundColor White
