#!/usr/bin/env bash
# ==============================================================================
# VoiceBridge — Debian / Ubuntu .deb Package Builder (amd64 & arm64 via dpkg-deb)
# Usage: ./packaging/linux/build_deb.sh <binary_path> <deb_arch> <output_deb_path> [version]
# ==============================================================================

set -euo pipefail

BIN_PATH="${1:?Usage: $0 <binary_path> <amd64|arm64> <output_deb_path> [version]}"
DEB_ARCH="${2:?Usage: $0 <binary_path> <amd64|arm64> <output_deb_path> [version]}"
OUT_DEB="${3:?Usage: $0 <binary_path> <amd64|arm64> <output_deb_path> [version]}"
RAW_VERSION="${4:-0.1.0}"
VERSION="${RAW_VERSION#v}"

if [ ! -f "$BIN_PATH" ]; then
    echo "Error: Binary not found at '$BIN_PATH'" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PKG_ROOT="$TMP_DIR/voicebridge_${VERSION}_${DEB_ARCH}"

mkdir -p "$PKG_ROOT/DEBIAN"
mkdir -p "$PKG_ROOT/usr/bin"
mkdir -p "$PKG_ROOT/usr/share/applications"
mkdir -p "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps"
mkdir -p "$PKG_ROOT/usr/share/pixmaps"
mkdir -p "$PKG_ROOT/usr/share/voicebridge/scripts"
mkdir -p "$PKG_ROOT/usr/share/doc/voicebridge"

# 1. Install executable & symlink
install -m 0755 "$BIN_PATH" "$PKG_ROOT/usr/bin/voicebridge"
ln -s voicebridge "$PKG_ROOT/usr/bin/VoiceBridge"

# 2. Install desktop entry, icons, scripts & docs
install -m 0644 packaging/linux/voicebridge.desktop "$PKG_ROOT/usr/share/applications/voicebridge.desktop"
install -m 0644 docs/logo.png "$PKG_ROOT/usr/share/icons/hicolor/256x256/apps/voicebridge.png"
install -m 0644 docs/logo.png "$PKG_ROOT/usr/share/pixmaps/voicebridge.png"
install -m 0644 config.example.json "$PKG_ROOT/usr/share/voicebridge/config.example.json"
install -m 0644 scripts/asr_server.py "$PKG_ROOT/usr/share/voicebridge/scripts/asr_server.py"
install -m 0755 scripts/start_local_asr.sh "$PKG_ROOT/usr/share/voicebridge/scripts/start_local_asr.sh"
install -m 0644 README.md "$PKG_ROOT/usr/share/doc/voicebridge/README.md"
install -m 0644 LICENSE "$PKG_ROOT/usr/share/doc/voicebridge/copyright"

INSTALLED_KB="$(du -sk "$PKG_ROOT/usr" | cut -f1)"

# 3. Generate DEBIAN/control
cat > "$PKG_ROOT/DEBIAN/control" <<EOF
Package: voicebridge
Version: ${VERSION}
Section: sound
Priority: optional
Architecture: ${DEB_ARCH}
Installed-Size: ${INSTALLED_KB}
Depends: libc6, libasound2t64 | libasound2, libxkbcommon0, libxkbcommon-x11-0, libxcb1, libx11-6, libx11-xcb1, libwayland-client0, libfontconfig1, libfreetype6, libdbus-1-3, libvulkan1
Recommends: python3
Maintainer: VoiceBridge Contributors <noreply@voicebridge.local>
Homepage: https://github.com/cclngit/VoiceBridge
Description: VoiceBridge — Real-Time Speech-To-Text & Multilingual Translation
 Ultra-fast offline real-time speech recognition and multilingual translation
 pipeline written in Rust, featuring a hardware-accelerated GPUI native desktop
 interface, Whisper local inference, and a transparent OBS Studio overlay.
EOF
chmod 0755 "$PKG_ROOT/DEBIAN"
chmod 0644 "$PKG_ROOT/DEBIAN/control"

# 4. Build .deb package via dpkg-deb
mkdir -p "$(dirname "$OUT_DEB")"
rm -f "$OUT_DEB"
dpkg-deb --build --root-owner-group "$PKG_ROOT" "$OUT_DEB"
dpkg-deb --info "$OUT_DEB"

echo "✅ Built Debian package: $OUT_DEB"

