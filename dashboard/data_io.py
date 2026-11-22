"""Read and validate the Invisible Oil Loss Detector's prepared dashboard outputs.

Purpose
-------
1. Keep file paths, column names and numeric conversion out of the UI code.
2. Load real outputs created by the existing Member 1 and Member 2 scripts.
3. Never invent missing wells, sensor measurements, failure events or predictions.
4. Treat ML scores as *relative ranking signals*, not calibrated probabilities.
5. Load well-level files only when the app explicitly enables LOCAL private mode.
6. Raise readable errors when a file has an incompatible schema.

Expected files (all private and outside public GitHub unless approved):
    outputs/dashboard_summary.json
    outputs/dashboard_daily.csv
    outputs/dashboard_loss_candidates.csv
    outputs/dashboard_well_recent.csv
    outputs/dashboard_failures.csv
    outputs/dashboard_well_areas.csv
    outputs/well_risk_predictions.csv
    outputs/maintenance_priorities.csv
The public app only reads dashboard/approved_public_summary.json if approved.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RESULTS = ROOT / "results"
DASH = ROOT / "dashboard"


def csv_or_empty(path: Path, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return a CSV with parsed dates, or an empty frame when not present."""
    path = Path(path)
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    for col in dates:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "uwi" in df:
        df["uwi"] = df["uwi"].astype("string").str.strip()
    return df


def json_or_empty(path: Path) -> dict:
    if not Path(path).is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def require_columns(df: pd.DataFrame, names: list[str], file: str) -> None:
    missing = sorted(set(names) - set(df.columns))
    if missing:
        raise ValueError(f"{file} missing required columns: {', '.join(missing)}")


def normalize_risks(df: pd.DataFrame) -> pd.DataFrame:
    """Adapt Member 2's real CSV. Do not interpret raw score as probability."""
    if df.empty:
        return df
    require_columns(df, ["uwi", "as_of_date", "risk_score"], "well_risk_predictions.csv")
    out = df.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    out["risk_score"] = pd.to_numeric(out["risk_score"], errors="coerce")
    out = out.dropna(subset=["uwi", "as_of_date", "risk_score"])
    out = out[out["risk_score"].between(0, 1)]
    out = out.sort_values("as_of_date").drop_duplicates("uwi", keep="last")
    # Risk percentile measures relative position within the loaded score set.
    out["risk_percentile"] = out["risk_score"].rank(pct=True).mul(100)
    return out


def normalize_priorities(df: pd.DataFrame) -> pd.DataFrame:
    """Load precomputed maintenance priorities, never generate invented ranks."""
    if df.empty:
        return df
    require_columns(df, ["uwi", "priority", "priority_score"], "maintenance_priorities.csv")
    out = df.copy()
    out["priority"] = out["priority"].astype("string").str.title()
    for col in ["priority_score", "risk_percentile", "risk_score", "daily_oil_at_risk_m3", "hypothetical_exposed_oil_m3"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    # Member 2 stores the risk rank as a 0–1 fraction; show a 0–100 percentile.
    if "risk_percentile" in out:
        out["risk_percentile"] = out["risk_percentile"] * 100
    out = out.sort_values("priority_score", ascending=False).drop_duplicates("uwi")
    return out


def join_priority_context(priorities: pd.DataFrame, areas: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Attach location category and anomaly evidence without multiplying rows."""
    out = priorities.copy()
    if out.empty:
        return out
    if not areas.empty and {"uwi", "field_operating_area"}.issubset(areas.columns):
        out = out.merge(areas[["uwi", "field_operating_area"]].drop_duplicates("uwi"), on="uwi", how="left", validate="one_to_one")
    if not candidates.empty and {"uwi", "candidate_days", "estimated_shortfall_m3"}.issubset(candidates.columns):
        out = out.merge(candidates[["uwi", "candidate_days", "estimated_shortfall_m3"]].drop_duplicates("uwi"), on="uwi", how="left", validate="one_to_one")
    return out


def load_private() -> dict:
    """Read private local files. Caller must guard access with private-mode flag."""
    daily = csv_or_empty(OUT / "dashboard_daily.csv", ("production_date",))
    recent = csv_or_empty(OUT / "dashboard_well_recent.csv", ("production_date",))
    candidates = csv_or_empty(OUT / "dashboard_loss_candidates.csv", ("last_candidate_date",))
    failures = csv_or_empty(OUT / "dashboard_failures.csv", ("failure_start_date",))
    areas = csv_or_empty(OUT / "dashboard_well_areas.csv")
    risks = normalize_risks(csv_or_empty(OUT / "well_risk_predictions.csv", ("as_of_date",)))
    priority = normalize_priorities(csv_or_empty(OUT / "maintenance_priorities.csv", ("as_of_date",)))
    priority = join_priority_context(priority, areas, candidates)
    return dict(
        summary=json_or_empty(OUT / "dashboard_summary.json"),
        daily=daily, recent=recent, candidates=candidates,
        failures=failures, areas=areas, risks=risks, priorities=priority,
        metrics=json_or_empty(RESULTS / "model_metrics.json"),
    )


def load_public() -> dict:
    """No private-file access: only reviewed public aggregate data and existing public model metrics."""
    return dict(
        summary=json_or_empty(DASH / "approved_public_summary.json"),
        metrics=json_or_empty(RESULTS / "model_metrics.json"),
    )
