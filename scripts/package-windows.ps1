[CmdletBinding()]
param(
    [switch] $SkipBuild,
    [string] $Destination
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($env:OS -ne "Windows_NT") {
    throw "This script must run on Windows. Use the Windows GitHub Actions workflow when building from macOS."
}

$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $repoRoot "dist"
} elseif (-not [System.IO.Path]::IsPathRooted($Destination)) {
    $Destination = Join-Path $repoRoot $Destination
}

$target = "x86_64-pc-windows-msvc"
$packageName = "GhostSun-Windows-x64"
$packageDir = Join-Path $Destination $packageName
$archive = Join-Path $Destination "$packageName.zip"
$originalRustFlags = [Environment]::GetEnvironmentVariable("RUSTFLAGS")

Push-Location $repoRoot
try {
    if (-not $SkipBuild) {
        rustup target add $target
        if ($LASTEXITCODE -ne 0) {
            throw "Could not install the Rust Windows target."
        }

        # Statically link the MSVC C runtime so the packaged app does not
        # depend on a separately installed Visual C++ Redistributable.
        if ([string]::IsNullOrWhiteSpace($originalRustFlags)) {
            $env:RUSTFLAGS = "-C target-feature=+crt-static"
        } else {
            $env:RUSTFLAGS = "$originalRustFlags -C target-feature=+crt-static"
        }
        cargo build --release --locked --package ghostsun-app --target $target
        if ($LASTEXITCODE -ne 0) {
            throw "The GhostSun Windows build failed."
        }
    }

    $executable = Join-Path $repoRoot "target\$target\release\ghostsun-app.exe"
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Windows executable not found at $executable. Build it first or omit -SkipBuild."
    }

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    if (Test-Path -LiteralPath $packageDir) {
        Remove-Item -LiteralPath $packageDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $packageDir | Out-Null

    Copy-Item -LiteralPath $executable -Destination (Join-Path $packageDir "GhostSun.exe")
    Copy-Item -LiteralPath (Join-Path $repoRoot "docs\windows.md") `
        -Destination (Join-Path $packageDir "README-Windows.md")

    $hash = (Get-FileHash -LiteralPath (Join-Path $packageDir "GhostSun.exe") -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  GhostSun.exe" | Set-Content -LiteralPath (Join-Path $packageDir "SHA256SUMS.txt") -Encoding ascii

    if (Test-Path -LiteralPath $archive) {
        Remove-Item -LiteralPath $archive -Force
    }
    Compress-Archive -Path (Join-Path $packageDir "*") -DestinationPath $archive

    Write-Host "Windows package: $archive"
    Write-Host "Executable SHA-256: $hash"
} finally {
    if ($null -eq $originalRustFlags) {
        Remove-Item Env:RUSTFLAGS -ErrorAction SilentlyContinue
    } else {
        $env:RUSTFLAGS = $originalRustFlags
    }
    Pop-Location
}
