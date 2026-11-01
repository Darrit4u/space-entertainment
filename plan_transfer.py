"""Offline circular estimates from the recorded snapshot; never connects to KSP."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from math_logic import OrbitalMath


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=root / "docs/experiments/transfer-inspection.json")
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are never overwritten")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    state = snapshot["inspection"]
    mu, r1, r2 = state["kerbin"]["mu_m3_s2"], state["orbit"]["semi_major_axis_m"], state["mun"]["semi_major_axis_m"]
    r, v, rm, vm = (state[k] for k in ("position_inertial_m", "velocity_inertial_m_s", "mun_position_inertial_m", "mun_velocity_inertial_m_s"))
    phase = OrbitalMath.current_phase_angle(r, rm, vm)
    inclination = OrbitalMath.relative_inclination(r, v, rm, vm)
    target_phase = OrbitalMath.hohmann_phase_angle(mu, r1, r2)
    omega_ship, omega_target = OrbitalMath.angular_velocity(mu, r1), OrbitalMath.angular_velocity(mu, r2)
    wait = OrbitalMath.time_to_phase_angle(target_phase, phase, omega_ship, omega_target)
    dv1, dv2 = OrbitalMath.hohmann_delta_v(mu, r1, r2)
    engine = next(e for e in state["engines"] if e["part"] == "liquidEngine3.v2")
    propellants = {p["name"]: p["reachable_units"] for p in engine["propellants"]}
    reachable = 100 * min(propellants["LiquidFuel"] / 9, propellants["Oxidizer"] / 11)
    reserve = 433.0  # Working landing reserve, fixed in transfer iteration 1.
    duration, fuel = OrbitalMath.burn_duration(dv1, state["mass_kg"], engine["max_vacuum_thrust_N"], engine["vacuum_specific_impulse_s"], max(0, reachable - reserve))
    if wait < duration / 2:
        wait += math.tau / abs(omega_ship - omega_target)
    result = {
        "baseline": snapshot["baseline"]["name"],
        "baseline_sha256": snapshot["baseline"]["sha256"],
        "input_snapshot_sha256_lf": hashlib.sha256(args.snapshot.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
        "scope": "Circular coplanar diagnostic estimate using the measured semi-major axes and target-plane projection; not an executable trajectory",
        "encounter_verified": False,
        "automatic_burn_authorized": False,
        "requires_spatial_encounter_prediction": inclination > 1e-7,
        "relative_inclination_deg": math.degrees(inclination),
        "projected_phase_deg": math.degrees(phase),
        "target_circular_phase_deg": math.degrees(target_phase),
        "circular_wait_estimate_s": wait,
        "candidate_node_ut": state["ut"] + wait,
        "candidate_burn_start_ut": state["ut"] + wait - duration / 2,
        "ideal_transfer_delta_v_m_s": dv1,
        "kerbin_circularization_delta_v_m_s_not_mun_capture": dv2,
        "ideal_transfer_duration_s": OrbitalMath.hohmann_transfer_time(mu, r1, r2),
        "constant_thrust_burn_duration_s": duration,
        "burn_propellant_kg": fuel,
        "engine_reachable_propellant_kg": reachable,
        "protected_central_propellant_kg": reserve,
        "central_propellant_after_burn_kg": reachable - fuel,
        "limitations": ["Projected phase has a nonconstant rate on an inclined orbit", "No three-dimensional encounter or Mun periapsis has been predicted", "Half-duration centering is a finite-burn approximation, not trajectory integration", "Mun capture and landing remain separate fuel expenditures"],
        "calculation_source_sha256_lf": {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for name in ("math_logic.py", "plan_transfer.py")},
    }
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
