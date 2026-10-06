#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-0.1.0}"
VERSION="${VERSION#v}" # Strip leading 'v' if present

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${DIST_DIR:-${ROOT_DIR}/target/dist}"

mkdir -p "${DIST_DIR}"

CLIENT_BIN="${CLIENT_BIN:-${ROOT_DIR}/target/release/telos}"
SERVER_BIN="${SERVER_BIN:-${ROOT_DIR}/target/release/telos-server}"

if [[ ! -f "${CLIENT_BIN}" ]] && [[ -f "${ROOT_DIR}/target/debug/telos" ]]; then
    CLIENT_BIN="${ROOT_DIR}/target/debug/telos"
fi
if [[ ! -f "${SERVER_BIN}" ]] && [[ -f "${ROOT_DIR}/target/debug/telos-server" ]]; then
    SERVER_BIN="${ROOT_DIR}/target/debug/telos-server"
fi

if [[ ! -f "${CLIENT_BIN}" ]] || [[ ! -f "${SERVER_BIN}" ]]; then
    echo "Error: Binaries not found. Build them first with:"
    echo "  cargo build --release --bin telos --bin telos-server"
    exit 1
fi

echo "==> Building Debian packages for Telos v${VERSION}..."

# 1. Build Client DEB
CLIENT_DEB_DIR="${ROOT_DIR}/target/deb/telos_${VERSION}_amd64"
rm -rf "${CLIENT_DEB_DIR}"
mkdir -p "${CLIENT_DEB_DIR}/DEBIAN"
mkdir -p "${CLIENT_DEB_DIR}/usr/bin"
mkdir -p "${CLIENT_DEB_DIR}/usr/share/applications"
mkdir -p "${CLIENT_DEB_DIR}/usr/share/pixmaps"
mkdir -p "${CLIENT_DEB_DIR}/usr/share/icons/hicolor/256x256/apps"
mkdir -p "${CLIENT_DEB_DIR}/usr/share/telos/assets"

install -m 755 "${CLIENT_BIN}" "${CLIENT_DEB_DIR}/usr/bin/telos"
install -m 644 "${SCRIPT_DIR}/telos.desktop" "${CLIENT_DEB_DIR}/usr/share/applications/telos.desktop"
install -m 644 "${SCRIPT_DIR}/telos.png" "${CLIENT_DEB_DIR}/usr/share/pixmaps/telos.png"
install -m 644 "${SCRIPT_DIR}/telos.png" "${CLIENT_DEB_DIR}/usr/share/icons/hicolor/256x256/apps/telos.png"
if [[ -d "${ROOT_DIR}/assets/telos" ]]; then
    cp -r "${ROOT_DIR}/assets/telos"/* "${CLIENT_DEB_DIR}/usr/share/telos/assets/" 2>/dev/null || true
fi

cat > "${CLIENT_DEB_DIR}/DEBIAN/control" << EOF
Package: telos
Version: ${VERSION}
Section: games
Priority: optional
Architecture: amd64
Maintainer: Larvance <64753457+larvance@users.noreply.github.com>
Depends: libc6, libvulkan1, libx11-6, libxkbcommon0
Description: High-performance voxel engine and game
 A world-class voxel engine and game in Rust and Vulkan.
 Features pixel-level LOD, tiny compressed chunks, GPU-driven rendering,
 and deterministic procedural terrain generation.
EOF

dpkg-deb --build --root-owner-group "${CLIENT_DEB_DIR}" "${DIST_DIR}/telos_${VERSION}_amd64.deb"
echo "Created: ${DIST_DIR}/telos_${VERSION}_amd64.deb"

# 2. Build Server DEB
SERVER_DEB_DIR="${ROOT_DIR}/target/deb/telos-server_${VERSION}_amd64"
rm -rf "${SERVER_DEB_DIR}"
mkdir -p "${SERVER_DEB_DIR}/DEBIAN"
mkdir -p "${SERVER_DEB_DIR}/usr/bin"
mkdir -p "${SERVER_DEB_DIR}/usr/lib/systemd/system"
mkdir -p "${SERVER_DEB_DIR}/etc/telos"
mkdir -p "${SERVER_DEB_DIR}/var/lib/telos"

install -m 755 "${SERVER_BIN}" "${SERVER_DEB_DIR}/usr/bin/telos-server"
install -m 644 "${SCRIPT_DIR}/telos-server.service" "${SERVER_DEB_DIR}/usr/lib/systemd/system/telos-server.service"
install -m 644 "${SCRIPT_DIR}/../server.toml" "${SERVER_DEB_DIR}/etc/telos/server.toml"

cat > "${SERVER_DEB_DIR}/DEBIAN/control" << EOF
Package: telos-server
Version: ${VERSION}
Section: games
Priority: optional
Architecture: amd64
Maintainer: Larvance <64753457+larvance@users.noreply.github.com>
Depends: libc6
Description: Voxel dedicated headless server
 Authoritative dedicated server for the Voxel game engine.
 Pure CPU, headless runtime with QUIC networking and multi-world simulation.
EOF

cat > "${SERVER_DEB_DIR}/DEBIAN/postinst" << 'EOF'
#!/bin/sh
set -e
if ! getent group telos >/dev/null; then
    groupadd --system telos
fi
if ! getent passwd telos >/dev/null; then
    useradd --system --ingroup telos --home /var/lib/telos --shell /usr/sbin/nologin telos
fi
chown -R telos:telos /var/lib/telos
exit 0
EOF
chmod 755 "${SERVER_DEB_DIR}/DEBIAN/postinst"

dpkg-deb --build --root-owner-group "${SERVER_DEB_DIR}" "${DIST_DIR}/telos-server_${VERSION}_amd64.deb"
echo "Created: ${DIST_DIR}/telos-server_${VERSION}_amd64.deb"

echo "==> Debian packages built successfully in ${DIST_DIR}"
