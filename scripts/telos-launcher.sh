#!/usr/bin/env bash
# ==============================================================================
# Telos Local Development Launcher
# Directly connected to local workspace at /home/usr/Projects/voxel
# Automatically ensures the latest build is executed.
# ==============================================================================
set -e

# Resolve repository directory (symlink-aware)
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

# Fallback if invoked detached
if [[ ! -f "$REPO_DIR/Cargo.toml" ]]; then
    REPO_DIR="/home/usr/Projects/larvance/telos"
fi

cd "$REPO_DIR"

TARGET_BIN="$REPO_DIR/target/release/telos"
if [[ "${TELOS_DEBUG:-0}" == "1" ]]; then
    TARGET_BIN="$REPO_DIR/target/debug/telos"
fi

# Rebuild incrementally if cargo is available and TELOS_NO_BUILD is not set
if [[ -z "${TELOS_NO_BUILD:-}" ]] && command -v cargo >/dev/null 2>&1; then
    if [ -t 1 ]; then
        if [[ "${TELOS_DEBUG:-0}" == "1" ]]; then
            cargo build --bin telos
        else
            cargo build --release --bin telos
        fi
    else
        # GUI launcher: build quietly in background
        if [[ "${TELOS_DEBUG:-0}" == "1" ]]; then
            cargo build --bin telos --quiet 2>/dev/null || true
        else
            cargo build --release --bin telos --quiet 2>/dev/null || true
        fi
    fi
fi

# Execute the latest binary with all passed arguments
if [[ -x "$TARGET_BIN" ]]; then
    exec "$TARGET_BIN" "$@"
elif [[ -x "$REPO_DIR/target/release/telos" ]]; then
    exec "$REPO_DIR/target/release/telos" "$@"
elif [[ -x "$REPO_DIR/target/debug/telos" ]]; then
    exec "$REPO_DIR/target/debug/telos" "$@"
else
    exec cargo run --release --bin telos -- "$@"
fi
