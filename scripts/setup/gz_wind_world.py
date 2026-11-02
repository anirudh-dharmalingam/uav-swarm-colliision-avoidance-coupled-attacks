#!/usr/bin/env python3
"""Build a wind-enabled copy of the PX4 Gazebo `default` world inside a container (never edits /opt).

Stock PX4 v1.17 gz assets do NOT apply wind to the X500: server.config does not load the WindEffects system
and x500_base/base_link has no <enable_wind>, so `windy.sdf` flies in still air.  This script writes, under
--out:
  worlds/default.sdf   default world (same name, so /world/default/* topics are unchanged) + <wind> mean velocity
  server.config        stock plugin list + gz::sim::systems::WindEffects (gust / meander / turbulence)
  models/x500_base/    copy of the stock model with <enable_wind>true</enable_wind> on base_link
Use: GZ_SIM_RESOURCE_PATH=<out>/models:$GZ_SIM_RESOURCE_PATH GZ_SIM_SERVER_CONFIG_PATH=<out>/server.config
     gz sim -s -r <out>/worlds/default.sdf

Wind model (gz WindEffects): mean horizontal wind `speed` m/s from `heading_deg` (ENU, direction the air moves
to), magnitude modulated by a sinusoidal gust (+-gust_pct, period gust_period_s) and white noise, direction
meander (+-meander_deg, 23 s), vertical white turbulence.  The link force is the plugin's linear-drag
approximation F = k * m * (v_wind - v_link); k is identified in docs/LE_PX4_WIND.md.
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

GZ = Path("/opt/PX4-Autopilot/Tools/simulation/gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", type=float, required=True)
    ap.add_argument("--heading-deg", type=float, default=0.0)
    ap.add_argument("--gust-pct", type=float, default=0.3)
    ap.add_argument("--gust-period-s", type=float, default=8.0)
    ap.add_argument("--meander-deg", type=float, default=15.0)
    ap.add_argument("--turb-std", type=float, default=0.3, help="white noise std of magnitude / vertical, m/s")
    ap.add_argument("--k", type=float, default=0.12)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "worlds").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)

    h = math.radians(a.heading_deg)
    vx, vy = a.speed * math.cos(h), a.speed * math.sin(h)
    w = (GZ / "worlds" / "default.sdf").read_text(encoding="utf-8-sig")
    assert "<wind>" not in w
    w = w.replace("</world>", f"    <wind>\n      <linear_velocity>{vx:.4f} {vy:.4f} 0</linear_velocity>\n    </wind>\n"
                              "  </world>", 1)
    (out / "worlds" / "default.sdf").write_text(w)

    plug = f"""    <plugin entity_name="*" entity_type="world" filename="gz-sim-wind-effects-system" name="gz::sim::systems::WindEffects">
      <force_approximation_scaling_factor>{a.k}</force_approximation_scaling_factor>
      <horizontal>
        <magnitude>
          <time_for_rise>5</time_for_rise>
          <sin><amplitude_percent>{a.gust_pct}</amplitude_percent><period>{a.gust_period_s}</period></sin>
          <noise type="gaussian"><mean>0</mean><stddev>{a.turb_std}</stddev></noise>
        </magnitude>
        <direction>
          <time_for_rise>5</time_for_rise>
          <sin><amplitude>{a.meander_deg}</amplitude><period>23</period></sin>
          <noise type="gaussian"><mean>0</mean><stddev>2</stddev></noise>
        </direction>
      </horizontal>
      <vertical>
        <noise type="gaussian"><mean>0</mean><stddev>{a.turb_std}</stddev></noise>
      </vertical>
    </plugin>
"""
    sc = (GZ / "server.config").read_text()
    sc = re.sub(r"(\s*</plugins>)", "\n" + plug + r"\1", sc, count=1)
    assert "WindEffects" in sc
    (out / "server.config").write_text(sc)

    dst = out / "models" / "x500_base"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(GZ / "models" / "x500_base", dst)
    m = (dst / "model.sdf").read_text()
    m2 = re.sub(r'(<link name="base_link">)', r"\1\n      <enable_wind>true</enable_wind>", m, count=1)
    assert m2 != m
    (dst / "model.sdf").write_text(m2)
    print(f"wind world: {vx:.2f},{vy:.2f} m/s k={a.k} gust={a.gust_pct} -> {out}")


if __name__ == "__main__":
    main()
