# src/extract_entities.py
import json
from pathlib import Path
import pandas as pd

# Input file (uploaded)
INPUT_PATH = Path("data/season_2526.json")  # <-- local path used as file URL by tooling

ENTITIES = [
    "players",
    "matchStatistics",
    "itemEvents",
    "itemPlayers",
    "itemPlayerHistory",
    "matchScoreStatistics",
]

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def to_dataframe(arr):
    """
    Turn a list into a DataFrame. If elements are dicts, json_normalize them
    to flatten nested dicts. Otherwise just make a DataFrame.
    """
    if not isinstance(arr, list):
        return pd.DataFrame([arr])  # single object -> single-row df
    if len(arr) == 0:
        return pd.DataFrame()
    if isinstance(arr[0], dict):
        return pd.json_normalize(arr, sep=".")
    else:
        return pd.DataFrame(arr)

def main():
    assert INPUT_PATH.exists(), f"Input file not found: {INPUT_PATH}"
    top = load_json(INPUT_PATH)
    resp = top.get("response", {})

    for ent in ENTITIES:
        arr = resp.get(ent, [])
        df = to_dataframe(arr)

        print(f"\n=== ENTITY: {ent} ===")
        print(f"Rows: {len(df)}")
        print(f"Columns (first 40): {list(df.columns)[:40]}")
        print(f"\nTop 10 rows for {ent}:")
        if len(df) > 0:
            # pretty print top 10 as table
            # limit column width for nicer display
            with pd.option_context('display.max_columns', None, 'display.width', 200):
                print(df.head(10).to_string(index=False))
        else:
            print("  <empty>")
        print("-" * 80)

    # Also show a small flattened players summary (no saving)
    if "players" in resp:
        players_arr = resp.get("players", [])
        players_df = to_dataframe(players_arr)
        candidate_cols = ["baseObjectId", "name", "excludeFromStatistics", "createdTimestamp", "updatedTimestamp"]
        selected = [c for c in candidate_cols if c in players_df.columns]
        if selected:
            players_flat = players_df[selected].rename(columns={
                "baseObjectId": "player_id",
                "createdTimestamp": "created_ts",
                "updatedTimestamp": "updated_ts",
                "excludeFromStatistics": "exclude_from_statistics"
            })
            print("\n=== players_flat sample ===")
            with pd.option_context('display.max_columns', None, 'display.width', 200):
                print(players_flat.head(10).to_string(index=False))
            print(f"Total players: {len(players_flat)}")
        else:
            print("\nplayers_flat: no expected columns found to flatten.")

if __name__ == "__main__":
    main()
