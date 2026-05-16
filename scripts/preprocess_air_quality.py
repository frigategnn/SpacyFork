import pandas as pd
import numpy as np
import torch
import os
import argparse

# ---------------------------------------------------------------------------
# Variate groups
# ---------------------------------------------------------------------------
# Pollutants: the "exposure" side of the causal graph
POLLUTANT_COLS = ["o3", "PM10", "PM2_5_DRY", "SO2", "NO", "NO2", "CO"]
POLLUTANT_COLS = ["PM10", "PM2_5_DRY"]

# Meteorological confounders: influence both pollution dispersion and health
METEO_COLS = ["RH", "T2"]
METEO_COLS = []

# Health outcomes: derived from DISEASE_CODE column via one-hot encoding
# Each row in the Excel is a patient visit with a single DISEASE_CODE.
# We count the number of visits per disease per grid cell per time period.
DISEASE_CODES = ["HTN", "CAD", "DBM", "COPD", "TB", "BA", "DEM", "LC"]
DISEASE_CODES = ["COPD", "TB"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess air-quality + health Excel data for use with SPACY."
    )
    parser.add_argument(
        "--input_file", type=str, default="../../Multi_pollutants_Scalling_input.xlsx"
    )
    parser.add_argument("--out_dir", type=str, default="../data/air_quality")
    parser.add_argument(
        "--nx", type=int, default=30, help="Horizontal (longitude) grid cells"
    )
    parser.add_argument(
        "--ny", type=int, default=30, help="Vertical (latitude) grid cells"
    )
    parser.add_argument(
        "--time_freq",
        type=str,
        default="W",
        choices=["D", "W", "M"],
        help="Time aggregation: D=Daily, W=Weekly, M=Monthly (default)",
    )
    parser.add_argument(
        "--include_confounders",
        action="store_true",
        help="Include age and gender as additional variates",
    )
    parser.add_argument(
        "--max_time",
        type=int,
        default=None,
        help="Max time steps to keep (for quick testing)",
    )
    return parser.parse_args()


def find_col(df, candidates):
    """Return the first candidate column name present in df (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def main():
    args = parse_args()
    print(f"Reading {args.input_file} (this may take a while for large files)...")

    try:
        df = pd.read_excel(args.input_file, engine="openpyxl")
    except Exception as e:
        print(f"Failed to read with openpyxl: {e}")
        return

    print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}\n")

    # ------------------------------------------------------------------
    # Required spatial + temporal columns
    # ------------------------------------------------------------------
    lat_col = find_col(df, ["Lat"])
    long_col = find_col(df, ["Long"])
    date_col = find_col(df, ["Date"])
    disease_col = find_col(df, ["DISEASE_CODE", "Diagnosis"])

    for name, col in [("Lat", lat_col), ("Long", long_col), ("Date", date_col)]:
        if col is None:
            print(f"Error: Missing required column '{name}'. Aborting.")
            return

    if disease_col is None:
        print(
            "Warning: No DISEASE_CODE or Diagnosis column found. Health outcomes will be missing!"
        )

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[long_col] = pd.to_numeric(df[long_col], errors="coerce")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[lat_col, long_col, date_col])
    print(f"After dropping NaN lat/long/date: {len(df)} rows remain.\n")

    # ------------------------------------------------------------------
    # One-hot encode DISEASE_CODE into separate binary columns
    # ------------------------------------------------------------------
    if disease_col is not None:
        print(f"Disease column: '{disease_col}'")
        print(f"Disease value counts:\n{df[disease_col].value_counts()}\n")

        # Create a binary column for each disease
        available_diseases = []
        for disease in DISEASE_CODES:
            col_name = f"disease_{disease}"
            df[col_name] = (
                df[disease_col].astype(str).str.strip().str.upper() == disease.upper()
            ).astype(int)
            count = df[col_name].sum()
            if count > 0:
                available_diseases.append(disease)
                print(f"  {disease}: {count} cases → column '{col_name}'")
            else:
                df.drop(columns=[col_name], inplace=True)
    else:
        available_diseases = []

    # ------------------------------------------------------------------
    # Spatial grid
    # ------------------------------------------------------------------
    min_lat, max_lat = df[lat_col].min(), df[lat_col].max()
    min_long, max_long = df[long_col].min(), df[long_col].max()

    print(f"\nSpatial bounds:")
    print(f"  Latitude:  [{min_lat:.4f}, {max_lat:.4f}]")
    print(f"  Longitude: [{min_long:.4f}, {max_long:.4f}]")

    lat_bins = np.linspace(min_lat, max_lat, args.ny + 1)
    long_bins = np.linspace(min_long, max_long, args.nx + 1)

    df["y_idx"] = np.digitize(df[lat_col], lat_bins) - 1
    df["x_idx"] = np.digitize(df[long_col], long_bins) - 1
    df["y_idx"] = df["y_idx"].clip(0, args.ny - 1)
    df["x_idx"] = df["x_idx"].clip(0, args.nx - 1)

    # ------------------------------------------------------------------
    # Time bins
    # ------------------------------------------------------------------
    df["TimePeriod"] = df[date_col].dt.to_period(args.time_freq)
    time_periods = sorted(df["TimePeriod"].unique())
    T = len(time_periods)
    time_map = {tp: i for i, tp in enumerate(time_periods)}
    df["t_idx"] = df["TimePeriod"].map(time_map)
    print(
        f"\nTime periods ({args.time_freq}): {T} steps, "
        f"from {time_periods[0]} to {time_periods[-1]}"
    )

    # Optionally truncate time for quick testing
    if args.max_time and args.max_time < T:
        T = args.max_time
        df = df[df["t_idx"] < T]
        print(f"  Truncated to T={T} for testing")

    # ------------------------------------------------------------------
    # Find available pollutant & meteo columns
    # ------------------------------------------------------------------
    available_pollutants = [c for c in POLLUTANT_COLS if find_col(df, [c]) is not None]
    available_meteo = [c for c in METEO_COLS if find_col(df, [c]) is not None]

    # ------------------------------------------------------------------
    # Individual confounders (age, gender)
    # ------------------------------------------------------------------
    age_col = find_col(df, ["Age", "age"])
    gender_col = find_col(df, ["Gender", "gender", "Sex", "sex"])
    use_confounders = args.include_confounders and (
        age_col is not None or gender_col is not None
    )

    # ------------------------------------------------------------------
    # Build full variate list
    # Ordering: pollutants | meteo | diseases | (confounders)
    # ------------------------------------------------------------------
    variate_names = (
        list(available_pollutants) + list(available_meteo) + list(available_diseases)
    )
    if use_confounders:
        if age_col is not None:
            variate_names.append("mean_age")
        if gender_col is not None:
            variate_names.append("prop_female")

    V = len(variate_names)
    N = args.ny * args.nx

    print(f"\nVariates ({V} total):")
    print(f"  Pollutants  ({len(available_pollutants)}): {available_pollutants}")
    print(f"  Meteorology ({len(available_meteo)}):      {available_meteo}")
    print(f"  Diseases    ({len(available_diseases)}):   {available_diseases}")
    if use_confounders:
        extras = []
        if age_col:
            extras.append("mean_age")
        if gender_col:
            extras.append("prop_female")
        print(f"  Confounders ({len(extras)}):              {extras}")
    print(f"\nTensor shape will be [{V}, {T}, {N}]")

    # ------------------------------------------------------------------
    # Build tensor  X[v, t, y, x]
    # ------------------------------------------------------------------
    X = np.zeros((V, T, args.ny, args.nx), dtype=np.float32)

    grouped = df.groupby(["t_idx", "y_idx", "x_idx"])

    # --- Pollutants + meteo: aggregate by MEAN (physical measurements) ---
    mean_cols = available_pollutants + available_meteo
    mean_actual = [find_col(df, [c]) for c in mean_cols]

    if mean_actual:
        mean_group = grouped[mean_actual].mean()
        for (t, y, x), row in mean_group.iterrows():
            if t >= T:
                continue
            for v_idx, actual_col in enumerate(mean_actual):
                val = row[actual_col]
                X[v_idx, t, y, x] = val if not np.isnan(val) else 0.0

    # --- Diseases: aggregate by SUM (count of cases per cell per period) ---
    disease_offset = len(mean_cols)
    disease_col_names = [f"disease_{d}" for d in available_diseases]

    if disease_col_names:
        disease_group = grouped[disease_col_names].sum()
        for (t, y, x), row in disease_group.iterrows():
            if t >= T:
                continue
            for i, col_name in enumerate(disease_col_names):
                X[disease_offset + i, t, y, x] = row[col_name]

    # --- Confounders ---
    if use_confounders:
        conf_offset = disease_offset + len(available_diseases)
        ci = 0
        if age_col is not None:
            df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
            age_group = grouped[age_col].mean()
            for (t, y, x), val in age_group.items():
                if t >= T:
                    continue
                X[conf_offset + ci, t, y, x] = val if not np.isnan(val) else 0.0
            ci += 1

        if gender_col is not None:
            df["_gender_bin"] = df[gender_col].map(
                lambda v: 1.0 if str(v).strip().upper() in ["F", "FEMALE", "1"] else 0.0
            )
            gender_group = grouped["_gender_bin"].mean()
            for (t, y, x), val in gender_group.items():
                if t >= T:
                    continue
                X[conf_offset + ci, t, y, x] = val if not np.isnan(val) else 0.0

    # ------------------------------------------------------------------
    # Z-score normalise each variate
    # ------------------------------------------------------------------
    for v in range(V):
        flat = X[v].ravel()
        mu, sigma = flat.mean(), flat.std()
        if sigma > 1e-8:
            X[v] = (X[v] - mu) / sigma

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    # # HACK: begin
    # T = 10
    # X = X[:, :T, :, :]  # (n_variates, n_timesteps, nx, ny)
    # # HACK: end
    X_tensor = torch.tensor(X).view(V, T, N)  # [V, T, num_nodes]

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(X_tensor, os.path.join(args.out_dir, "0.pt"))
    print(f"\nSaved X tensor  shape={tuple(X_tensor.shape)}  →  {args.out_dir}/0.pt")

    # Placeholder graph
    G = torch.zeros((N, N))
    torch.save(G, os.path.join(args.out_dir, "graph.pt"))
    print(f"Saved G tensor  shape={tuple(G.shape)}  →  {args.out_dir}/graph.pt")

    # ------------------------------------------------------------------
    # Print config summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Update configs/data/air_quality.yaml with:")
    print("=" * 60)
    print(f"  nx:           {args.nx}")
    print(f"  ny:           {args.ny}")
    print(f"  num_nodes:    <nearest multiple of {V} to {N}>")
    print(f"  num_variates: {V}")
    print(f"  time_length:  {T}")
    print(f"  lat_bounds:   [{min_lat:.4f}, {max_lat:.4f}]")
    print(f"  long_bounds:  [{min_long:.4f}, {max_long:.4f}]")
    print("=" * 60)
    print("\nVariate index mapping (for interpreting the causal graph):")
    for i, name in enumerate(variate_names):
        if name in available_pollutants:
            group = "POLLUTANT"
        elif name in available_meteo:
            group = "METEO"
        elif name in available_diseases:
            group = "DISEASE"
        else:
            group = "CONFOUNDER"
        print(f"  [{i:2d}] {name:20s}  ({group})")

    # Suggest num_nodes
    for candidate in range(N, N + V):
        if candidate % V == 0:
            print(
                f"\n  Recommended num_nodes: {candidate}  ({candidate} = {V} × {candidate // V})"
            )
            break

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
