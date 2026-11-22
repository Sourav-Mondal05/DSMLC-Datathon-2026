
from pathlib import Path

import pandas as pd


INPUT = Path("outputs/production_features.parquet")
OUTPUT = Path("outputs/persistent_anomaly_summary.csv")


def main():
    print("Loading engineered production features...")

    df = pd.read_parquet(
        INPUT,
        columns=[
            "uwi",
            "production_date",
            "possible_invisible_loss",
            "potential_oil_loss_m3",
        ],
    )

    df["production_date"] = pd.to_datetime(df["production_date"])
    df = df.sort_values(["uwi", "production_date"])

    # Keep only candidate anomaly records.
    alerts = df.loc[df["possible_invisible_loss"].fillna(False)].copy()

    print(f"Total candidate alert days: {len(alerts):,}")
    print(f"Affected wells: {alerts['uwi'].nunique():,}")

    if alerts.empty:
        print("No candidate anomalies found.")
        return

    # Start a new episode when the well changes or the dates
    # are not consecutive calendar days.
    day_gap = alerts.groupby("uwi")["production_date"].diff().dt.days

    new_episode = day_gap.ne(1)
    alerts["episode_id"] = new_episode.cumsum()

    # Each episode represents consecutive candidate alert days.
    episodes = (
        alerts.groupby(["uwi", "episode_id"], as_index=False)
        .agg(
            start_date=("production_date", "min"),
            end_date=("production_date", "max"),
            duration_days=("production_date", "size"),
            estimated_shortfall_m3=("potential_oil_loss_m3", "sum"),
        )
    )

    episodes["persistent_3_days"] = episodes["duration_days"] >= 3
    episodes["persistent_7_days"] = episodes["duration_days"] >= 7

    total_episodes = len(episodes)
    three_day = int(episodes["persistent_3_days"].sum())
    seven_day = int(episodes["persistent_7_days"].sum())

    print("\nPERSISTENT ANOMALY ANALYSIS")
    print(f"Total anomaly episodes: {total_episodes:,}")
    print(f"Episodes lasting at least 3 days: {three_day:,}")
    print(f"Episodes lasting at least 7 days: {seven_day:,}")
    print(f"Longest episode: {episodes['duration_days'].max():,} days")

    print("\nTOP 10 LONGEST EPISODES")
    print(
        episodes.nlargest(10, "duration_days")[
            [
                "uwi",
                "start_date",
                "end_date",
                "duration_days",
                "estimated_shortfall_m3",
            ]
        ].to_string(index=False)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(OUTPUT, index=False)

    print(f"\nSaved: {OUTPUT}")
    print("Persistent anomaly analysis completed.")
    print("Note: Persistence does not confirm actual recoverable oil loss.")


if __name__ == "__main__":
    main()
