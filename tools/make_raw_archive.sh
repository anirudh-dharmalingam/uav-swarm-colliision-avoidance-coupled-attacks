#!/usr/bin/env bash
# Pack the raw traces that are NOT in the git bundle (per-agent PX4 telemetry, process logs, PX4 interrupted/failed attempts,
# Layer-1 per-episode truth traces) for an archival deposit (Zenodo / institutional repository / GitHub release assets).
# Run it against the ORIGINAL research repository, not the bundle:
#
#     bash tools/make_raw_archive.sh /path/to/Disaster-Swarm-dev  /path/to/output_dir
#
# Produces one tar.zst per campaign (each < 2 GB, the GitHub release-asset limit) and SHA256SUMS.  Needs tar with zstd.
# Approximate input sizes: PX4 confirm 270 MB, PX4 dev+dev2 150 MB, Layer-1 per-episode traces 2.3 GB.
set -euo pipefail
SRC=$(readlink -f "${1:?source repository}")
OUT=$(readlink -f "${2:?output directory}")
mkdir -p "$OUT"
cd "$SRC"
pack() {  # name, paths...
  local name=$1; shift
  echo "[..] $name"
  tar --zstd -cf "$OUT/$name.tar.zst" --sort=name --owner=0 --group=0 --numeric-owner --mtime='2026-11-01 00:00Z' "$@"
  ls -lh "$OUT/$name.tar.zst"
}
pack manuscriptB_px4_confirm results/rarf_hocbf_v1/confirm results/rarf_hocbf_v1/confirm_launch.log results/rarf_hocbf_v1/chain.log
pack manuscriptB_px4_dev     results/rarf_hocbf_v1/dev results/rarf_hocbf_v1/dev2 results/rarf_hocbf_v1/dev_launch.log results/rarf_hocbf_v1/dev2_launch.log
pack manuscriptB_l1_spectrum results/rarf_l1/breadth_spectrum
pack manuscriptB_l1_budget   results/rarf_l1/breadth_budget
pack manuscriptB_l1_replays  results/rarf_l1/hocbf_confirm_replay results/rarf_l1/v3_confirm_replay
(cd "$OUT" && sha256sum manuscriptB_*.tar.zst > SHA256SUMS && cat SHA256SUMS)
echo "[OK] upload $OUT/*.tar.zst and SHA256SUMS; cite the DOI in the bundle README (Data availability)."
