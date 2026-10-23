"""Run the planner on a sample orbit: an ISS TLE and a small AOI near its ground track."""
from datetime import datetime, timedelta, timezone

import final_submission as planner

TLE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005"
TLE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815350  4012"
START = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
END = START + timedelta(minutes=10)

# Place the AOI under the satellite's ground track 5 minutes into the pass,
# shifted east by `offset_deg` to make the pointing problem harder.
from sgp4.api import Satrec
sat = Satrec.twoline2rv(TLE1, TLE2)
mid = START + timedelta(minutes=5)
r, _ = planner._sat_state(sat, mid)
lat, lon = planner._ecef_to_llh_deg(planner._eci_to_ecef(r, planner._gmst(mid)))

for name, offset_deg in [("near nadir", 0.0), ("off track", 4.0)]:
    c = lon + offset_deg
    aoi = [(lat - 0.3, c - 0.3), (lat - 0.3, c + 0.3), (lat + 0.3, c + 0.3),
           (lat + 0.3, c - 0.3), (lat - 0.3, c - 0.3)]
    plan = planner.plan_imaging(TLE1, TLE2, aoi, START.isoformat(), END.isoformat(),
                                {"fov_deg": [2.0, 2.0]})
    print(f"{name:>10}: {len(plan['shutter'])} frames, {len(plan['attitude'])} attitude points")
    print(f"            {plan['notes']}")
