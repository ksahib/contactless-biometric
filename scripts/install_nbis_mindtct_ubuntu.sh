#!/usr/bin/env bash
set -euo pipefail

# Install only the NBIS components needed for MINDTCT on Ubuntu/WSL.
# Defaults:
#   source:  ~/src/Rel_5.0.0
#   install: ~/opt/nbis/bin/mindtct
#
# Optional overrides:
#   NBIS_VERSION=5_0_0
#   NBIS_URL=https://nigos.nist.gov/nist/nbis/nbis_v5_0_0.zip
#   SRC_ROOT=$HOME/src
#   INSTALL_PREFIX=$HOME/opt/nbis

NBIS_VERSION="${NBIS_VERSION:-5_0_0}"
NBIS_URL="${NBIS_URL:-https://nigos.nist.gov/nist/nbis/nbis_v${NBIS_VERSION}.zip}"
SRC_ROOT="${SRC_ROOT:-$HOME/src}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/opt/nbis}"
ZIP_PATH="$SRC_ROOT/nbis_v${NBIS_VERSION}.zip"

need_cmd() {
  command -v "$1" >/dev/null 2>&1
}

echo "[1/7] Installing Ubuntu build dependencies"
sudo apt update
sudo apt install -y build-essential make perl unzip curl cmake

mkdir -p "$SRC_ROOT" "$INSTALL_PREFIX/bin" "$INSTALL_PREFIX/lib" "$INSTALL_PREFIX/include"

echo "[2/7] Downloading NBIS source"
if [[ ! -f "$ZIP_PATH" ]]; then
  curl -L -o "$ZIP_PATH" "$NBIS_URL"
else
  echo "Using existing $ZIP_PATH"
fi

echo "[3/7] Extracting NBIS source"
cd "$SRC_ROOT"
unzip -q -o "$ZIP_PATH"

NBIS_SRC="$(find "$SRC_ROOT" -maxdepth 2 -type f -name setup.sh -printf '%h\n' | sort | head -n 1)"
if [[ -z "$NBIS_SRC" ]]; then
  echo "error: setup.sh not found after extracting $ZIP_PATH" >&2
  exit 1
fi

echo "Using NBIS source: $NBIS_SRC"
cd "$NBIS_SRC"

echo "[4/7] Fixing ownership/permissions for current user"
sudo chown -R "$USER:$USER" "$NBIS_SRC" "$INSTALL_PREFIX"
chmod -R u+rwX "$NBIS_SRC" "$INSTALL_PREFIX"

echo "[5/7] Configuring NBIS"
./setup.sh "$INSTALL_PREFIX" --without-X11
NBIS_CONFIG_PACKAGES="${NBIS_CONFIG_PACKAGES:-ijg png openjp2 commonnbis an2k imgtools mindtct pcasys}"
NBIS_BUILD_PACKAGES="${NBIS_BUILD_PACKAGES:-ijg png openjp2 commonnbis an2k imgtools mindtct}"
make PACKAGES="$NBIS_CONFIG_PACKAGES" config

echo "[6/7] Building commonnbis and mindtct"
for package in $NBIS_CONFIG_PACKAGES; do
  make -C "$package" cpheaders
done
for package in $NBIS_BUILD_PACKAGES; do
  make -C "$package" libs
done
make -C mindtct bins

MINDTCT_BUILT="$NBIS_SRC/mindtct/bin/mindtct"
if [[ ! -x "$MINDTCT_BUILT" ]]; then
  echo "error: expected built mindtct at $MINDTCT_BUILT" >&2
  exit 1
fi

echo "[7/7] Installing mindtct to $INSTALL_PREFIX/bin"
cp "$MINDTCT_BUILT" "$INSTALL_PREFIX/bin/mindtct"
chmod +x "$INSTALL_PREFIX/bin/mindtct"

if ! grep -qs 'opt/nbis/bin' "$HOME/.bashrc"; then
  printf '\nexport PATH="$HOME/opt/nbis/bin:$PATH"\n' >> "$HOME/.bashrc"
fi

echo
echo "Installed:"
"$INSTALL_PREFIX/bin/mindtct" 2>&1 | head -n 3 || true
echo
echo "Add to current shell now:"
echo "  export PATH=\"$INSTALL_PREFIX/bin:\$PATH\""
echo
echo "Verify:"
echo "  which mindtct"
echo "  mindtct"
