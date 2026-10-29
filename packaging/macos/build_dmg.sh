#!/usr/bin/env bash
# ==============================================================================
# VoiceBridge — macOS .app Bundle & .dmg Disk Image Builder (ARM64 & Universal2)
# Usage: ./packaging/macos/build_dmg.sh <binary_path> <output_dmg_path> [version]
# ==============================================================================

set -euo pipefail

BIN_PATH="${1:?Usage: $0 <binary_path> <output_dmg_path> [version]}"
OUT_DMG="${2:?Usage: $0 <binary_path> <output_dmg_path> [version]}"
RAW_VERSION="${3:-0.1.0}"
VERSION="${RAW_VERSION#v}"

if [ ! -f "$BIN_PATH" ]; then
    echo "Error: Binary not found at '$BIN_PATH'" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

APP_DIR="$TMP_DIR/stage/VoiceBridge.app"
ICONSET_DIR="$TMP_DIR/AppIcon.iconset"

mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources/scripts"
mkdir -p "$APP_DIR/Contents/Resources/docs"
mkdir -p "$ICONSET_DIR"

# 1. Generate AppIcon.icns from docs/logo.png
sips -z 16 16     docs/logo.png --out "$ICONSET_DIR/icon_16x16.png"      >/dev/null
sips -z 32 32     docs/logo.png --out "$ICONSET_DIR/icon_16x16@2x.png"   >/dev/null
sips -z 32 32     docs/logo.png --out "$ICONSET_DIR/icon_32x32.png"      >/dev/null
sips -z 64 64     docs/logo.png --out "$ICONSET_DIR/icon_32x32@2x.png"   >/dev/null
sips -z 128 128   docs/logo.png --out "$ICONSET_DIR/icon_128x128.png"    >/dev/null
sips -z 256 256   docs/logo.png --out "$ICONSET_DIR/icon_128x128@2x.png" >/dev/null
sips -z 256 256   docs/logo.png --out "$ICONSET_DIR/icon_256x256.png"    >/dev/null
sips -z 512 512   docs/logo.png --out "$ICONSET_DIR/icon_256x256@2x.png" >/dev/null
sips -z 512 512   docs/logo.png --out "$ICONSET_DIR/icon_512x512.png"    >/dev/null
sips -z 1024 1024 docs/logo.png --out "$ICONSET_DIR/icon_512x512@2x.png" >/dev/null
iconutil -c icns "$ICONSET_DIR" -o "$APP_DIR/Contents/Resources/AppIcon.icns"

# 2. Populate VoiceBridge.app bundle
sed "s/__VERSION__/$VERSION/g" packaging/macos/Info.plist > "$APP_DIR/Contents/Info.plist"
cp "$BIN_PATH" "$APP_DIR/Contents/MacOS/VoiceBridge"
chmod 755 "$APP_DIR/Contents/MacOS/VoiceBridge"

cp config.example.json "$APP_DIR/Contents/Resources/"
cp scripts/asr_server.py scripts/start_local_asr.sh "$APP_DIR/Contents/Resources/scripts/"
chmod 755 "$APP_DIR/Contents/Resources/scripts/start_local_asr.sh"
cp docs/logo.png "$APP_DIR/Contents/Resources/docs/"

# 3. Ad-hoc sign the .app bundle (required on Apple Silicon macOS)
codesign --force --deep --sign - "$APP_DIR"

# 4. Add Applications drag-and-drop symlink & build compressed UDZO .dmg
ln -s /Applications "$TMP_DIR/stage/Applications"
mkdir -p "$(dirname "$OUT_DMG")"
rm -f "$OUT_DMG"
hdiutil create \
    -volname "VoiceBridge" \
    -srcfolder "$TMP_DIR/stage" \
    -ov \
    -format UDZO \
    "$OUT_DMG" >/dev/null

echo "✅ Built macOS DMG: $OUT_DMG ($(du -h "$OUT_DMG" | cut -f1))"

