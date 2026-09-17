"""Plot a fused latitude/longitude CSV trajectory on an interactive Folium map."""

import argparse
import webbrowser
from pathlib import Path

import folium
import pandas as pd


def plot_path(csv_path: Path, output_html: Path, open_browser: bool = False) -> Path:
    df = pd.read_csv(csv_path)
    coords_df = df[["lat_out", "lon_out"]].dropna()
    if coords_df.empty:
        raise RuntimeError("No valid lat_out/lon_out rows were found.")

    coords = list(zip(coords_df["lat_out"], coords_df["lon_out"]))
    map_view = folium.Map(location=coords[0], zoom_start=18)
    folium.PolyLine(coords, color="red", weight=3).add_to(map_view)
    folium.Marker(coords[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(map_view)
    folium.Marker(coords[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(map_view)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    map_view.save(str(output_html))
    if open_browser:
        webbrowser.open(output_html.resolve().as_uri())
    return output_html


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="CSV containing lat_out and lon_out columns")
    parser.add_argument("--output", type=Path, default=Path("outputs/uav_path.html"))
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args()
    saved = plot_path(args.csv, args.output, args.open_browser)
    print(f"Map saved to: {saved}")


if __name__ == "__main__":
    main()
