#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-0.1.0}"
VERSION="${VERSION#v}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${ROOT_DIR}/target/dist"

mkdir -p "${DIST_DIR}"

CLIENT_BIN="${ROOT_DIR}/target/release/voxel"

if [[ ! -f "${CLIENT_BIN}" ]] && [[ -f "${ROOT_DIR}/target/debug/voxel" ]]; then
    CLIENT_BIN="${ROOT_DIR}/target/debug/voxel"
fi

if [[ ! -f "${CLIENT_BIN}" ]]; then
    echo "Error: Binary not found at ${CLIENT_BIN}."
    echo "Build it first with: cargo build --bin voxel"
    exit 1
fi

echo "==> Building Linux AppImage for Voxel v${VERSION}..."

APPDIR="${ROOT_DIR}/target/appdir/Voxel.AppDir"
rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/usr/bin"
mkdir -p "${APPDIR}/usr/share/voxel/assets"
mkdir -p "${APPDIR}/usr/share/icons/hicolor/256x256/apps"

install -m 755 "${CLIENT_BIN}" "${APPDIR}/usr/bin/voxel"
install -m 644 "${SCRIPT_DIR}/voxel.desktop" "${APPDIR}/voxel.desktop"
install -m 644 "${SCRIPT_DIR}/voxel.png" "${APPDIR}/voxel.png"
install -m 644 "${SCRIPT_DIR}/voxel.png" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/voxel.png"

if [[ -d "${ROOT_DIR}/assets/voxel" ]]; then
    cp -r "${ROOT_DIR}/assets/voxel"/* "${APPDIR}/usr/share/voxel/assets/" 2>/dev/null || true
fi

cat > "${APPDIR}/AppRun" << 'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "${0}")")"
export PATH="${HERE}/usr/bin:${PATH}"
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH:-}"
export VOXEL_ASSETS_DIR="${HERE}/usr/share/voxel/assets"
exec "${HERE}/usr/bin/voxel" "$@"
EOF
chmod 755 "${APPDIR}/AppRun"

# Check if appimagetool is available, or download it
TOOL="${ROOT_DIR}/target/appimagetool"
if command -v appimagetool >/dev/null 2>&1; then
    TOOL="$(command -v appimagetool)"
elif [[ ! -f "${TOOL}" ]]; then
    echo "Downloading appimagetool..."
    curl -sSL -o "${TOOL}" "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage" || \
    curl -sSL -o "${TOOL}" "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
    chmod +x "${TOOL}"
fi

OUTPUT_APPIMAGE="${DIST_DIR}/voxel-${VERSION}-x86_64.AppImage"
echo "Running appimagetool..."
ARCH=x86_64 "${TOOL}" --appimage-extract-and-run "${APPDIR}" "${OUTPUT_APPIMAGE}" 2>/dev/null || \
ARCH=x86_64 "${TOOL}" "${APPDIR}" "${OUTPUT_APPIMAGE}"

echo "Created: ${OUTPUT_APPIMAGE}"
