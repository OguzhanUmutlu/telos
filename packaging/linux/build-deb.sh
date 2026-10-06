#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-0.1.0}"
VERSION="${VERSION#v}" # Strip leading 'v' if present

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DIST_DIR="${DIST_DIR:-${ROOT_DIR}/target/dist}"

mkdir -p "${DIST_DIR}"

CLIENT_BIN="${CLIENT_BIN:-${ROOT_DIR}/target/release/voxel}"
SERVER_BIN="${SERVER_BIN:-${ROOT_DIR}/target/release/voxel-server}"

if [[ ! -f "${CLIENT_BIN}" ]] && [[ -f "${ROOT_DIR}/target/debug/voxel" ]]; then
    CLIENT_BIN="${ROOT_DIR}/target/debug/voxel"
fi
if [[ ! -f "${SERVER_BIN}" ]] && [[ -f "${ROOT_DIR}/target/debug/voxel-server" ]]; then
    SERVER_BIN="${ROOT_DIR}/target/debug/voxel-server"
fi

if [[ ! -f "${CLIENT_BIN}" ]] || [[ ! -f "${SERVER_BIN}" ]]; then
    echo "Error: Binaries not found. Build them first with:"
    echo "  cargo build --release --bin voxel --bin voxel-server"
    exit 1
fi

echo "==> Building Debian packages for Voxel v${VERSION}..."

# 1. Build Client DEB
CLIENT_DEB_DIR="${ROOT_DIR}/target/deb/voxel_${VERSION}_amd64"
rm -rf "${CLIENT_DEB_DIR}"
mkdir -p "${CLIENT_DEB_DIR}/DEBIAN"
mkdir -p "${CLIENT_DEB_DIR}/usr/bin"
mkdir -p "${CLIENT_DEB_DIR}/usr/share/applications"
mkdir -p "${CLIENT_DEB_DIR}/usr/share/pixmaps"
mkdir -p "${CLIENT_DEB_DIR}/usr/share/icons/hicolor/256x256/apps"
mkdir -p "${CLIENT_DEB_DIR}/usr/share/voxel/assets"

install -m 755 "${CLIENT_BIN}" "${CLIENT_DEB_DIR}/usr/bin/voxel"
install -m 644 "${SCRIPT_DIR}/voxel.desktop" "${CLIENT_DEB_DIR}/usr/share/applications/voxel.desktop"
install -m 644 "${SCRIPT_DIR}/voxel.png" "${CLIENT_DEB_DIR}/usr/share/pixmaps/voxel.png"
install -m 644 "${SCRIPT_DIR}/voxel.png" "${CLIENT_DEB_DIR}/usr/share/icons/hicolor/256x256/apps/voxel.png"
if [[ -d "${ROOT_DIR}/assets/voxel" ]]; then
    cp -r "${ROOT_DIR}/assets/voxel"/* "${CLIENT_DEB_DIR}/usr/share/voxel/assets/" 2>/dev/null || true
fi

cat > "${CLIENT_DEB_DIR}/DEBIAN/control" << EOF
Package: voxel
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

dpkg-deb --build --root-owner-group "${CLIENT_DEB_DIR}" "${DIST_DIR}/voxel_${VERSION}_amd64.deb"
echo "Created: ${DIST_DIR}/voxel_${VERSION}_amd64.deb"

# 2. Build Server DEB
SERVER_DEB_DIR="${ROOT_DIR}/target/deb/voxel-server_${VERSION}_amd64"
rm -rf "${SERVER_DEB_DIR}"
mkdir -p "${SERVER_DEB_DIR}/DEBIAN"
mkdir -p "${SERVER_DEB_DIR}/usr/bin"
mkdir -p "${SERVER_DEB_DIR}/usr/lib/systemd/system"
mkdir -p "${SERVER_DEB_DIR}/etc/voxel"
mkdir -p "${SERVER_DEB_DIR}/var/lib/voxel"

install -m 755 "${SERVER_BIN}" "${SERVER_DEB_DIR}/usr/bin/voxel-server"
install -m 644 "${SCRIPT_DIR}/voxel-server.service" "${SERVER_DEB_DIR}/usr/lib/systemd/system/voxel-server.service"
install -m 644 "${SCRIPT_DIR}/../server.toml" "${SERVER_DEB_DIR}/etc/voxel/server.toml"

cat > "${SERVER_DEB_DIR}/DEBIAN/control" << EOF
Package: voxel-server
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
if ! getent group voxel >/dev/null; then
    groupadd --system voxel
fi
if ! getent passwd voxel >/dev/null; then
    useradd --system --ingroup voxel --home /var/lib/voxel --shell /usr/sbin/nologin voxel
fi
chown -R voxel:voxel /var/lib/voxel
exit 0
EOF
chmod 755 "${SERVER_DEB_DIR}/DEBIAN/postinst"

dpkg-deb --build --root-owner-group "${SERVER_DEB_DIR}" "${DIST_DIR}/voxel-server_${VERSION}_amd64.deb"
echo "Created: ${DIST_DIR}/voxel-server_${VERSION}_amd64.deb"

echo "==> Debian packages built successfully in ${DIST_DIR}"
