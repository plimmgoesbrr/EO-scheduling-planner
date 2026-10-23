# EO Scheduling Planner

A constraint-aware imaging planner for an Earth-observation satellite. Given the satellite's orbit, a target area on the ground and a pass window, it produces a complete pointing schedule: when to turn, where to point, and when to fire the camera, so that the photos cover as much of the target area as possible without breaking the spacecraft's hardware limits.

Built for the **418 Hackathon** (inter-university), 2026.

## The problem

A satellite flying overhead has only a few minutes to photograph a target. Its camera sees a small 2° × 2° patch, so it has to turn (slew) between shots. The difficulty is in the limits:

| Constraint | Limit used |
|---|---|
| Off-nadir angle (how far from straight down it may look) | ≤ 53° for easy/medium cases, ≤ 55° for the hard case |
| Body rate during exposure (the camera must be still while the 0.12 s shutter is open) | target 0.03 °/s |
| Reaction-wheel momentum (how much turning the wheels can absorb) | target 0.025 N·m·s |

## How it works

1. **Orbit propagation:** the satellite's position and velocity over the pass are computed from its TLE with **SGP4**, the standard orbit model.
2. **Geometry:** the target polygon is converted between geographic (lat/lon), Earth-fixed (ECEF) and inertial (ECI) frames on the **WGS-84** ellipsoid.
3. **Difficulty check:** the planner finds the moment the satellite is closest to looking straight at the target and picks a strategy:
    - **Easy / medium (target ≤ 45° off-nadir):** *stop-and-stare.* A grid of aim points over the target, visited in snake order to keep slews short, each scored on slew size, off-nadir angle and timing.
    - **Hard (target far off-track):** *footprint grazing.* The camera's centre stays within the 55° limit, but the edge or corner of its field of view is allowed to clip the target. Candidate pointings are searched over a cone of directions and roll angles.
4. **Footprint projection:** each candidate pointing is turned into the actual patch of ground it sees by intersecting the four field-of-view corner rays with the Earth's ellipsoid.
5. **Coverage scoring:** each footprint is clipped against the target polygon (Sutherland–Hodgman) to measure how much new area it covers.
6. **Schedule build:** attitudes are expressed as **quaternions**, interpolated with **SLERP**, and padded with hold periods around every exposure so the spacecraft is still when the shutter opens.

## Usage

```bash
pip install -r requirements.txt
python example.py
```

`example.py` plans two passes on a sample orbit (ISS TLE): one target near the ground track and one 4° off-track.

To use your own inputs:

```python
from final_submission import plan_imaging

plan = plan_imaging(
    tle_line1, tle_line2,
    aoi_polygon_llh,                 # [(lat, lon), ...] closed polygon, degrees
    "2024-01-01T12:00:00Z",          # pass start (UTC)
    "2024-01-01T12:10:00Z",          # pass end (UTC)
    {"fov_deg": [2.0, 2.0]},         # spacecraft parameters
)
plan["attitude"]   # [{"t": seconds, "q_BN": [x, y, z, w]}, ...]  pointing over time
plan["shutter"]    # [{"t_start": seconds, "duration": 0.12}, ...]  camera exposures
plan["notes"]      # strategy chosen and limits applied
```

## Stack

Python, NumPy, SGP4
