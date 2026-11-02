#!/usr/bin/env bash
# Build PX4 SITL once and keep it across `docker run --rm` containers.
#
# /opt/PX4-Autopilot/build lives inside the container and is lost when the
# container exits. We redirect it into the mounted workspace so the build
# (and ccache) survive. Safe to re-run: it is a no-op when the build exists.
set -euo pipefail

PX4_DIR="${PX4_DIR:-/opt/PX4-Autopilot}"
CACHE_ROOT="${DRONELAB_CACHE:-/workspace/.cache}"
BUILD_CACHE="${CACHE_ROOT}/px4_build"
export CCACHE_DIR="${CACHE_ROOT}/ccache"

mkdir -p "${BUILD_CACHE}" "${CCACHE_DIR}"

if [ -d "${PX4_DIR}/build" ] && [ ! -L "${PX4_DIR}/build" ]; then
    # A real build dir from this container: move it into the cache.
    rsync -a "${PX4_DIR}/build/" "${BUILD_CACHE}/"
    rm -rf "${PX4_DIR}/build"
fi
ln -sfn "${BUILD_CACHE}" "${PX4_DIR}/build"

if [ -x "${PX4_DIR}/build/px4_sitl_default/bin/px4" ] && [ "${1:-}" != "--force" ]; then
    echo "[OK] PX4 SITL build present: ${BUILD_CACHE}/px4_sitl_default"
    exit 0
fi

echo "[..] Building PX4 SITL (first time takes ~5-15 min)"
cd "${PX4_DIR}"
make px4_sitl_default
echo "[OK] PX4 SITL built"
