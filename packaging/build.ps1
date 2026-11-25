# Build the distributable zip for the version in scripts\app_info.py:
#   dist\v<version>\RiceSeedCounter\   and   dist\RiceSeedCounter_v<version>.zip
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
# Earlier builds (other versions, and the old dist\RiceSeedCounter) are left untouched.
# 1) PyInstaller folder build  2) put the guide, LICENSE and licenses\ next to the exe
# 3) drop the PC-specific GPU cache  4) zip
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = "$root\riceseedcounter-env\python.exe"

$ver = (& $py -c "import sys; sys.path.insert(0, r'$root\scripts'); import app_info; print(app_info.VERSION)").Trim()
$distpath = "$root\dist\v$ver"
$d = "$distpath\RiceSeedCounter"
$zip = "$root\dist\RiceSeedCounter_v$ver.zip"
"Building version $ver -> $d"

# Rebuilding the same version: remove its previous output first. Running the built exe leaves
# read-only GPU cache files that PyInstaller cannot delete; a running exe also locks the folder.
if (Get-Process RiceSeedCounter -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$d*" }) {
    throw "Close $d\RiceSeedCounter.exe first"
}
if (Test-Path $d) {
    Get-ChildItem $d -Recurse -Force -File | ForEach-Object { $_.IsReadOnly = $false }
    Remove-Item $d -Recurse -Force
}

& $py -m PyInstaller RiceSeedCounter.spec --noconfirm --distpath $distpath --workpath "$root\build"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# user guide (the Korean file name is matched by wildcard: PowerShell 5.1 reads this script as ANSI)
Copy-Item "$root\packaging\*.txt" $d -Force
Copy-Item "$root\LICENSE.txt" $d -Force
Copy-Item "$root\licenses" "$d\licenses" -Recurse -Force

# compiled GPU kernels are driver-specific and read-only; each PC builds its own
Get-ChildItem $d -Recurse -Filter *.blob | ForEach-Object { $_.IsReadOnly = $false; Remove-Item $_.FullName -Force }
Remove-Item "$d\results" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$root\build" -Recurse -Force -ErrorAction SilentlyContinue

if (Test-Path $zip) { Remove-Item $zip }
Set-Location $distpath
& "$env:SystemRoot\System32\tar.exe" -a -c -f $zip RiceSeedCounter
Set-Location $root
"Done: $zip ({0:N0} MB)" -f ((Get-Item $zip).Length / 1MB)
