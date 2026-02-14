#!/usr/bin/env bash
# ubuntu_install.sh - prepare GDAL + python venv and common packages on Ubuntu
# Usage: sudo ./ubuntu_install.sh [-p venv_path] [-r requirements.txt] [-y]
set -euo pipefail

VENV_PATH="./venv"
REQ_FILE="./requirements.txt"
ASSUME_YES=0

while getopts "p:r:y" opt; do
    case "$opt" in
        p) VENV_PATH="$OPTARG" ;;
        r) REQ_FILE="$OPTARG" ;;
        y) ASSUME_YES=1 ;;
        *) echo "Usage: $0 [-p venv_path] [-r requirements.txt] [-y]"; exit 1 ;;
    esac
done

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        echo "Run as root or install sudo."; exit 1
    fi
fi

export DEBIAN_FRONTEND=noninteractive

echo "Updating apt and installing system packages..."
if [ "$ASSUME_YES" -eq 1 ]; then
    $SUDO apt-get update -yq
    $SUDO apt-get install -yq build-essential python3-dev python3-venv python3-pip \
        libgdal-dev gdal-bin libproj-dev proj-bin libgeos-dev curl wget git
else
    $SUDO apt-get update
    $SUDO apt-get install build-essential python3-dev python3-venv python3-pip \
        libgdal-dev gdal-bin libproj-dev proj-bin libgeos-dev curl wget git
fi

PYTHON=$(command -v python3)
if [ -z "$PYTHON" ]; then
    echo "python3 not found"; exit 1
fi

echo "Creating python virtualenv at: $VENV_PATH"
if [ ! -d "$VENV_PATH" ]; then
    $PYTHON -m venv "$VENV_PATH"
fi

# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"

pip install --upgrade pip setuptools wheel

# If gdal-config present, try to pin GDAL Python package to system version
if command -v gdal-config >/dev/null 2>&1; then
    GDAL_VER=$(gdal-config --version)
    echo "Found gdal-config, version $GDAL_VER. Installing matching GDAL Python package..."
    pip install --no-cache-dir "GDAL==${GDAL_VER}"
fi

# If a requirements file exists, install it; otherwise install common geospatial packages
if [ -f "$REQ_FILE" ]; then
    echo "Installing Python packages from $REQ_FILE"
    pip install --no-cache-dir -r "$REQ_FILE"
else
    echo "Installing common Python packages (numpy, pandas, rasterio, fiona, shapely, pyproj, requests)"
    pip install --no-cache-dir numpy pandas rasterio fiona shapely pyproj requests
fi

deactivate

echo "Setup complete. Activate the venv with: source $VENV_PATH/bin/activate"
# Ensure venv isn't accidentally committed, and save installed packages if no requirements file exists
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GITIGNORE="$(git rev-parse --show-toplevel)/.gitignore"
    IGNORE_ENTRY="${VENV_PATH%/}"
    if [ ! -f "$GITIGNORE" ] || ! grep -Fxq "$IGNORE_ENTRY" "$GITIGNORE"; then
        printf "\n# local python venv\n%s\n" "$IGNORE_ENTRY" >> "$GITIGNORE"
        echo "Added $IGNORE_ENTRY to $GITIGNORE"
    fi
fi

if [ ! -f "$REQ_FILE" ]; then
    if [ -x "$VENV_PATH/bin/pip" ]; then
        echo "Saving current venv packages to $REQ_FILE"
        "$VENV_PATH/bin/pip" freeze > "$REQ_FILE" || true
    fi
fi