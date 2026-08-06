param(
    [string]$LocalWheel = ""
)

# Maintainer pin configuration. Run scripts/update_oneclick_pins.py before release.
$UvVersion = "0.12.2"
$UvUrl = "https://github.com/astral-sh/uv/releases/download/0.12.2/uv-x86_64-pc-windows-msvc.zip"
$UvSha256 = "01442d8ce5c7124151a73e697c836d252c6da853c18c73206d3cc4c2378a91d2"
$PythonVersion = "3.12.10"
$LatestReleaseApiUrl = "https://api.github.com/repos/daraskme/Modern-IOpaint/releases/latest"
$ReleasesPageUrl = "https://github.com/daraskme/Modern-IOpaint/releases"
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$InstallRoot = $PSScriptRoot
$ToolsDir = Join-Path $InstallRoot "tools"
$EnvDir = Join-Path $InstallRoot "env"
$LogPath = Join-Path $InstallRoot "setup.log"
$UvArchive = Join-Path $ToolsDir "uv-windows.zip"
$UvExe = Join-Path $ToolsDir "uv.exe"
$PythonExe = Join-Path $EnvDir "Scripts\python.exe"
$ModernIOPaintExe = Join-Path $EnvDir "Scripts\modern-iopaint.exe"
$MinimumFreeBytes = 45GB
$ScriptExitCode = 0

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Description,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host "`n==> $Description" -ForegroundColor Cyan
    $ErrorActionPreference = "Continue"
    & $FilePath @Arguments
    $ProcessExitCode = $LASTEXITCODE
    if ($ProcessExitCode -ne 0) {
        throw "$Description failed with exit code $ProcessExitCode."
    }
}

try {
    Start-Transcript -Path $LogPath -Append | Out-Null
    Write-Host "Modern-IOPaint Windows bootstrap" -ForegroundColor Green
    Write-Host "Transcript: $LogPath"

    if (
        $UvVersion.Contains("PLACEHOLDER") -or
        $UvUrl.Contains("PLACEHOLDER") -or
        $UvSha256.Contains("PLACEHOLDER")
    ) {
        throw "uv is not pinned. The maintainer must run scripts/update_oneclick_pins.py before distributing this installer."
    }
    if ($UvSha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw "The configured uv archive SHA-256 is not a 64-character hexadecimal digest."
    }

    $NvidiaCommand = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    $NvidiaSmi = if ($NvidiaCommand) { $NvidiaCommand.Source } else { $null }
    if (-not $NvidiaSmi) {
        throw "No NVIDIA GPU/driver was found: nvidia-smi.exe is unavailable. Install or update the NVIDIA driver first."
    }

    $GpuRows = @(& $NvidiaSmi --query-gpu=name,driver_version,compute_cap --format=csv,noheader,nounits 2>&1)
    if ($LASTEXITCODE -ne 0) {
        $GpuRows = @(& $NvidiaSmi --query-gpu=name,driver_version --format=csv,noheader,nounits 2>&1)
    }
    if ($LASTEXITCODE -ne 0 -or $GpuRows.Count -eq 0) {
        throw "nvidia-smi could not query an NVIDIA GPU: $($GpuRows -join ' ')"
    }
    Write-Host "NVIDIA GPU(s):"
    $GpuRows | ForEach-Object { Write-Host "  $_" }

    $SmiSummary = @(& $NvidiaSmi 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "nvidia-smi failed while checking CUDA driver support."
    }
    $CudaMatch = [regex]::Match(($SmiSummary -join "`n"), 'CUDA Version:\s*(\d+)\.(\d+)')
    if ($CudaMatch.Success) {
        $CudaMajor = [int]$CudaMatch.Groups[1].Value
        $CudaMinor = [int]$CudaMatch.Groups[2].Value
        if ($CudaMajor -lt 12 -or ($CudaMajor -eq 12 -and $CudaMinor -lt 8)) {
            throw "The NVIDIA driver reports CUDA $CudaMajor.$CudaMinor support. CUDA 12.8 or newer is required; update the driver."
        }
        Write-Host "Driver CUDA support: $CudaMajor.$CudaMinor"
    }
    else {
        Write-Warning "nvidia-smi did not report its supported CUDA version; setup-gpu will perform the final driver check."
    }

    $InstallDriveName = (Get-Item -LiteralPath $InstallRoot).PSDrive.Name
    $InstallDrive = Get-PSDrive -Name $InstallDriveName
    if ($InstallDrive.Free -lt $MinimumFreeBytes) {
        $FreeGiB = [math]::Round($InstallDrive.Free / 1GB, 1)
        throw "At least 45 GB free disk space is required; only $FreeGiB GB is free. The application/dependencies use about 4 GB and model downloads use about 12-40 GB depending on selection."
    }
    Write-Host "Disk preflight passed: $([math]::Round($InstallDrive.Free / 1GB, 1)) GB free."

    New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
    $NeedUvDownload = $true
    if (Test-Path -LiteralPath $UvArchive) {
        $ExistingHash = (Get-FileHash -LiteralPath $UvArchive -Algorithm SHA256).Hash.ToLowerInvariant()
        $NeedUvDownload = $ExistingHash -ne $UvSha256.ToLowerInvariant()
    }
    if ($NeedUvDownload) {
        Write-Host "Downloading pinned uv $UvVersion from $UvUrl"
        Invoke-WebRequest -UseBasicParsing -Uri $UvUrl -OutFile $UvArchive
    }
    $DownloadedHash = (Get-FileHash -LiteralPath $UvArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($DownloadedHash -ne $UvSha256.ToLowerInvariant()) {
        Remove-Item -LiteralPath $UvArchive -Force
        throw "uv archive SHA-256 mismatch. The untrusted download was removed."
    }

    $UvExtractDir = Join-Path $ToolsDir "uv-$UvVersion"
    New-Item -ItemType Directory -Force -Path $UvExtractDir | Out-Null
    Expand-Archive -LiteralPath $UvArchive -DestinationPath $UvExtractDir -Force
    $ExtractedUv = Get-ChildItem -LiteralPath $UvExtractDir -Filter "uv.exe" -File -Recurse | Select-Object -First 1
    if (-not $ExtractedUv) {
        throw "The verified uv archive does not contain uv.exe."
    }
    Copy-Item -LiteralPath $ExtractedUv.FullName -Destination $UvExe -Force
    $env:PATH = "$ToolsDir;$env:PATH"

    Invoke-Checked "Installing uv-managed Python $PythonVersion" $UvExe @("python", "install", $PythonVersion)

    $CreateEnvironment = -not (Test-Path -LiteralPath $PythonExe)
    if (-not $CreateEnvironment) {
        $ExistingPythonOutput = @(& $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null)
        $PythonProbeExitCode = $LASTEXITCODE
        $ExistingPython = ($ExistingPythonOutput -join "").Trim()
        if ($PythonProbeExitCode -ne 0 -or $ExistingPython -ne "3.12") {
            Invoke-Checked "Repairing the virtual environment with Python $PythonVersion" $UvExe @("venv", "--python", $PythonVersion, "--clear", $EnvDir)
        }
    }
    else {
        Invoke-Checked "Creating the virtual environment with Python $PythonVersion" $UvExe @("venv", "--python", $PythonVersion, $EnvDir)
    }

    if ($LocalWheel) {
        if (-not (Test-Path -LiteralPath $LocalWheel -PathType Leaf)) {
            throw "The -LocalWheel file does not exist: $LocalWheel"
        }
        $ResolvedWheel = (Resolve-Path -LiteralPath $LocalWheel).Path
        Invoke-Checked "Installing Modern-IOPaint from local wheel" $UvExe @("pip", "install", "--python", $PythonExe, "--upgrade", "--force-reinstall", $ResolvedWheel)
    }
    else {
        Write-Host "Retrieving the latest Modern-IOPaint GitHub Release"
        try {
            $LatestRelease = Invoke-RestMethod -UseBasicParsing -Uri $LatestReleaseApiUrl -Headers @{ Accept = "application/vnd.github+json" }
        }
        catch {
            throw "Unable to retrieve the latest Modern-IOPaint GitHub Release. Ensure a published release exists at $ReleasesPageUrl and retry. The release must include a modern_iopaint-*.whl asset. Details: $($_.Exception.Message)"
        }

        $WheelAsset = @(
            $LatestRelease.assets | Where-Object { $_.name -like "modern_iopaint-*.whl" }
        ) | Select-Object -First 1
        if (-not $WheelAsset) {
            throw "The latest Modern-IOPaint GitHub Release does not contain a modern_iopaint-*.whl asset. Attach the built wheel to the release at $ReleasesPageUrl and retry."
        }

        $ReleaseWheel = Join-Path $ToolsDir ([System.IO.Path]::GetFileName([string]$WheelAsset.name))
        Write-Host "Downloading $($WheelAsset.name) from the latest GitHub Release"
        Invoke-WebRequest -UseBasicParsing -Uri ([string]$WheelAsset.browser_download_url) -OutFile $ReleaseWheel
        Invoke-Checked "Installing/updating Modern-IOPaint from GitHub Release" $UvExe @("pip", "install", "--python", $PythonExe, "--upgrade", $ReleaseWheel)
    }

    Invoke-Checked "Installing torch CUDA 12.8 runtime" $UvExe @("pip", "install", "--python", $PythonExe, "--torch-backend=cu128", "torch~=2.11.0", "torchvision~=0.26.0")
    Invoke-Checked "Installing and verifying Nunchaku" $ModernIOPaintExe @("setup-gpu")
    Invoke-Checked "Launching Modern-IOPaint" $ModernIOPaintExe @("start", "--model", "qwen-image", "--port", "8080", "--inbrowser")
}
catch {
    $ScriptExitCode = 1
    Write-Error $_ -ErrorAction Continue
}
finally {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
        # Start-Transcript can fail before it creates a transcript; preserve the real error.
    }
}

exit $ScriptExitCode
