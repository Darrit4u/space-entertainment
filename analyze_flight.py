import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).parent


def load_data(filename=None):
    if filename is None:
        logs = sorted((PROJECT_DIR / "logs").glob("flight_*.csv"))
        filename = logs[-1] if logs else PROJECT_DIR / "flight_log.csv"
    df = pd.read_csv(filename)
    if df.empty:
        raise ValueError(f"No telemetry rows in {filename}")
    print(f"Telemetry: {filename}")
    return df


def plot_altitude(df):
    plt.figure()
    plt.plot(df["t"], df["altitude"])
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [m]")
    plt.title("Altitude vs Time")
    plt.grid()
    plt.show()


def plot_speed(df):
    plt.figure()
    plt.plot(df["t"], df["speed"], label="Total speed")
    plt.plot(df["t"], df["vertical_speed"], label="Vertical speed")
    if "horizontal_speed" in df:
        plt.plot(df["t"], df["horizontal_speed"], label="Horizontal surface speed")
    if "target_vertical_speed" in df:
        plt.plot(df["t"], df["target_vertical_speed"], label="Target vertical speed", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.title("Speed vs Time")
    plt.legend()
    plt.grid()
    plt.show()


def plot_mass(df):
    plt.figure()
    plt.plot(df["t"], df["mass"])
    plt.xlabel("Time [s]")
    plt.ylabel("Mass [kg]")
    plt.title("Mass vs Time")
    plt.grid()
    plt.show()


def plot_orbit(df):
    plt.figure()
    plt.plot(df["t"], df["apoapsis"], label="Apoapsis")
    plt.plot(df["t"], df["periapsis"], label="Periapsis")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [m]")
    plt.title("Orbit evolution")
    plt.legend()
    plt.grid()
    plt.show()


def summary_table(df):
    summary = {
        "Max altitude [m]": df["altitude"].max(),
        "Max speed [m/s]": df["speed"].max(),
        "Total delta-v [m/s]": df["delta_v"].iloc[-1],
        "Final mass [kg]": df["mass"].iloc[-1],
        "Final vertical speed [m/s]": df["vertical_speed"].iloc[-1],
        "Flight time [s]": df["t"].iloc[-1],
    }

    table = pd.DataFrame(summary, index=["Value"])
    return table.T


def split_by_body(df):
    return dict(tuple(df.groupby("body")))


def main():
    parser = argparse.ArgumentParser(description="Plot a flight telemetry CSV")
    parser.add_argument("filename", nargs="?", help="CSV path; defaults to the latest run")
    args = parser.parse_args()
    try:
        df = load_data(args.filename)
    except (OSError, ValueError, pd.errors.EmptyDataError) as error:
        parser.exit(1, f"Cannot load telemetry: {error}\n")

    # --- Графики ---
    plot_altitude(df)
    plot_speed(df)
    plot_mass(df)
    plot_orbit(df)

    # --- Таблица ---
    table = summary_table(df)
    print("\n=== SUMMARY TABLE ===\n")
    print(table)

    # --- Разделение по SOI ---
    bodies = split_by_body(df)
    for body, part in bodies.items():
        print(f"\n--- {body} ---")
        print(f"Duration: {part['t'].iloc[-1] - part['t'].iloc[0]:.1f} s")
        print(f"Δv: {part['delta_v'].iloc[-1] - part['delta_v'].iloc[0]:.1f} m/s")


if __name__ == "__main__":
    main()
