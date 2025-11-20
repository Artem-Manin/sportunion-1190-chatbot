import json
import pandas as pd

path = "data/season_2526.json"

# Load JSON
with open(path, "r", encoding="utf-8") as f:
    j = json.load(f)

# Access correct nested structure
players_json = j["response"]["players"]

players = []
for p in players_json:
    players.append({
        "player_id": p.get("baseObjectId"),
        "name": p.get("name", "").strip(),
        "exclude_from_statistics": bool(p.get("excludeFromStatistics", False)),
        "created_ts": p.get("createdTimestamp"),
        "updated_ts": p.get("updatedTimestamp"),
    })

players_df = pd.DataFrame(players)

print(players_df)
