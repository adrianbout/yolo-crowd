# Create a clean release tar file
# This script excludes large folders to keep the archive small

$version = "v1.0.9"
$outputFile = "chaircounter-$version.tar"

Write-Host "Creating release archive: $outputFile" -ForegroundColor Green

# Create a temporary directory for the clean copy
$tempDir = "temp_release"
if (Test-Path $tempDir) {
    Remove-Item -Recurse -Force $tempDir
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

Write-Host "Copying files..." -ForegroundColor Yellow

# Copy all files and folders EXCEPT the excluded ones
$excludeFolders = @(
    ".venv",
    "venv",
    ".git",
    "runs",
    "videos",
    "backup",
    ".vscode",
    ".claude",
    "temp_release"
)

$excludeFiles = @(
    "*.tar",
    "*.mp4",
    "*.log",
    "nul",
    "run.txt",
    "test_api.py",
    "setup_bypass.txt"
)

# Copy directory structure
Get-ChildItem -Path . -Recurse | ForEach-Object {
    $relativePath = $_.FullName.Substring((Get-Location).Path.Length + 1)
    
    # Check if path contains any excluded folder
    $shouldExclude = $false
    foreach ($excludeFolder in $excludeFolders) {
        if ($relativePath -like "*$excludeFolder*") {
            $shouldExclude = $true
            break
        }
    }
    
    # Check if file matches excluded patterns
    foreach ($pattern in $excludeFiles) {
        if ($_.Name -like $pattern) {
            $shouldExclude = $true
            break
        }
    }
    
    # Skip __pycache__ folders
    if ($relativePath -like "*__pycache__*") {
        $shouldExclude = $true
    }
    
    if (-not $shouldExclude) {
        $destPath = Join-Path $tempDir $relativePath
        
        if ($_.PSIsContainer) {
            if (-not (Test-Path $destPath)) {
                New-Item -ItemType Directory -Path $destPath -Force | Out-Null
            }
        } else {
            $destDir = Split-Path $destPath -Parent
            if (-not (Test-Path $destDir)) {
                New-Item -ItemType Directory -Path $destDir -Force | Out-Null
            }
            Copy-Item $_.FullName -Destination $destPath -Force
        }
    }
}

Write-Host "Creating tar archive..." -ForegroundColor Yellow

# Create tar file from temp directory
tar -cf $outputFile -C $tempDir .

# Clean up temp directory
Remove-Item -Recurse -Force $tempDir

# Show file size
$fileSize = (Get-Item $outputFile).Length / 1MB
Write-Host "`nRelease created successfully!" -ForegroundColor Green
Write-Host "File: $outputFile" -ForegroundColor Cyan
Write-Host "Size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Cyan

Write-Host "`nExcluded folders:" -ForegroundColor Yellow
$excludeFolders | ForEach-Object { Write-Host "  - $_" }
