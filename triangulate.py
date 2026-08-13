""" Triangulate and coordinates """

import pandas as pd

# TDOA threshold (ms) for inclusion in triangulation
TDOA_THRESHOLD = 0.025

# Read the CSV files and combine them into a single DataFrame
files = [
    "pheasant_results_sd1.csv",
    "pheasant_results_sd2.csv",
    "pheasant_results_sd3.csv"
]

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["source_file"] = f
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
print(f"Rows before tdoa filter: {len(combined_df)}")

# Filter rows based on TDOA threshold
combined_df = combined_df[
    combined_df["tdoa_ms"].abs() >= TDOA_THRESHOLD
]
print(f"Rows remaining: {len(combined_df)}")