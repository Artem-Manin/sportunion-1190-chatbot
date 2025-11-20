# src/build_player_stats_and_matches.py
import json
from pathlib import Path
import pandas as pd

# IMPORTANT: use this project-local path (will be transformed to a URL by your tooling)
INPUT_PATH = Path("data/season_2526.json")

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def to_df(obj):
    """Convert list/dict to pandas DataFrame, flattening nested dicts."""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, list):
        if len(obj) == 0:
            return pd.DataFrame()
        if isinstance(obj[0], dict):
            return pd.json_normalize(obj, sep='.')
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        return pd.json_normalize([obj], sep='.')
    return pd.DataFrame([obj])

# -------------------------
# Build players dataframe (including exclude_from_statistics)
# -------------------------
def build_players_df(resp):
    players = resp.get("players", [])
    df = to_df(players)

    mapping = {
        "baseObjectId": "player_id",
        "name": "name",
        "excludeFromStatistics": "exclude_from_statistics",
        "createdTimestamp": "created_ts",
        "updatedTimestamp": "updated_ts",
    }
    present = [c for c in mapping.keys() if c in df.columns]
    if present:
        players_df = df[present].rename(columns=mapping)
    else:
        # consistent empty frame
        players_df = pd.DataFrame(columns=list(mapping.values()))

    # types & defaults
    if "player_id" in players_df.columns:
        players_df["player_id"] = pd.to_numeric(players_df["player_id"], errors="coerce").astype("Int64")
    else:
        players_df["player_id"] = pd.Series(dtype="Int64")

    if "name" not in players_df.columns:
        players_df["name"] = ""
    players_df["name"] = players_df["name"].fillna("").astype(str)

    if "exclude_from_statistics" in players_df.columns:
        players_df["exclude_from_statistics"] = players_df["exclude_from_statistics"].fillna(False).astype(bool)
    else:
        players_df["exclude_from_statistics"] = False

    if "created_ts" in players_df.columns:
        players_df["created_ts"] = pd.to_numeric(players_df["created_ts"], errors="coerce").astype("Int64")
    if "updated_ts" in players_df.columns:
        players_df["updated_ts"] = pd.to_numeric(players_df["updated_ts"], errors="coerce").astype("Int64")

    return players_df.reset_index(drop=True)

# -------------------------
# Aggregate goals/assists/own goals from matchStatistics
# -------------------------
def build_stats_from_matchstatistics(resp):
    ms = to_df(resp.get("matchStatistics", []))
    if ms.empty:
        return pd.DataFrame(columns=["player_id","goals","assists","own_goals"])

    # find player id column
    player_col = None
    for c in ("playerId","player_id","playerID"):
        if c in ms.columns:
            player_col = c
            break
    if player_col is None:
        return pd.DataFrame(columns=["player_id","goals","assists","own_goals"])

    ms[player_col] = pd.to_numeric(ms[player_col], errors="coerce").astype("Int64")

    def pick_flag(df, names):
        for n in names:
            if n in df.columns:
                return pd.to_numeric(df[n], errors="coerce").fillna(0).astype(int)
        return pd.Series(0, index=df.index, dtype=int)

    goals = pick_flag(ms, ["score", "isGoal", "Score"])
    assists = pick_flag(ms, ["assist", "isAssist", "Assist"])
    own_goals = pick_flag(ms, ["ownGoal", "own_goal", "OwnGoal"])

    ms["_pid"] = ms[player_col]

    agg = ms.groupby("_pid").agg(
        goals = ("_pid", lambda s: int(goals.loc[s.index].sum())),
        assists = ("_pid", lambda s: int(assists.loc[s.index].sum())),
        own_goals = ("_pid", lambda s: int(own_goals.loc[s.index].sum()))
    ).reset_index().rename(columns={"_pid":"player_id"})

    agg["player_id"] = pd.to_numeric(agg["player_id"], errors="coerce").astype("Int64")
    return agg

# -------------------------
# Matches played per player from itemPlayerHistory
# -------------------------
def build_matches_played_from_itemplayerhistory(resp):
    iph = to_df(resp.get("itemPlayerHistory", []))
    if iph.empty:
        return pd.DataFrame(columns=["player_id","matches_played"])

    # find player id column
    player_col = None
    for c in ("playerId","player_id","playerID"):
        if c in iph.columns:
            player_col = c
            break
    if player_col is None:
        return pd.DataFrame(columns=["player_id","matches_played"])

    iph["player_id"] = pd.to_numeric(iph[player_col], errors="coerce").astype("Int64")

    # derive a match identifier: prefer soccerMatchId when present and non-zero, else fallback to itemEventId
    if "soccerMatchId" in iph.columns:
        iph["soccer_match_id"] = pd.to_numeric(iph["soccerMatchId"], errors="coerce").astype("Int64")
    else:
        iph["soccer_match_id"] = pd.NA

    if "itemEventId" in iph.columns:
        iph["item_event_id_str"] = iph["itemEventId"].astype(str).fillna("")
    else:
        iph["item_event_id_str"] = ""

    def choose_match_id(row):
        sm = row.get("soccer_match_id")
        try:
            if pd.notna(sm) and int(sm) != 0:
                return f"s{int(sm)}"
        except Exception:
            pass
        return f"ie{row.get('item_event_id_str','')}"
    iph["match_identifier"] = iph.apply(choose_match_id, axis=1)

    mp = iph.groupby("player_id")["match_identifier"].nunique().reset_index().rename(columns={"match_identifier":"matches_played"})
    mp["matches_played"] = mp["matches_played"].astype(int)

    return mp

# -------------------------
# Build matches_df (one row per match) from matchScoreStatistics
# -------------------------
def build_matches_df(resp):
    mss = to_df(resp.get("matchScoreStatistics", []))
    if mss.empty:
        return pd.DataFrame()

    # ensure relevant columns exist
    for c in ["soccerMatchId","itemEventId","teamHome","teamAway","scoreTeamHome","scoreTeamAway","itemEventDate","teamHomeId","teamAwayId"]:
        if c not in mss.columns:
            mss[c] = pd.NA

    # create deterministic match_key: use soccerMatchId when present and non-zero, else composite
    def make_match_key(row):
        sm = row.get("soccerMatchId")
        try:
            if pd.notna(sm) and int(sm) != 0:
                return f"s{int(sm)}"
        except Exception:
            pass
        ie = row.get("itemEventId") or ""
        th = row.get("teamHomeId") or ""
        ta = row.get("teamAwayId") or ""
        date = row.get("itemEventDate") or ""
        return f"ie{ie}_h{th}_a{ta}_d{date}"

    mss["match_key"] = mss.apply(make_match_key, axis=1)

    # select only the necessary columns to avoid groupby.apply future warning
    columns_needed = [
        "match_key",
        "soccerMatchId","itemEventId","itemEventDate",
        "teamHome","teamAway","teamHomeId","teamAwayId",
        "scoreTeamHome","scoreTeamAway"
    ]
    subset = mss[columns_needed].copy()

    def final_for_group(g):
        try:
            home_scores = pd.to_numeric(g["scoreTeamHome"], errors="coerce")
            away_scores = pd.to_numeric(g["scoreTeamAway"], errors="coerce")
            home_final = int(home_scores.dropna().max()) if not home_scores.dropna().empty else None
            away_final = int(away_scores.dropna().max()) if not away_scores.dropna().empty else None
        except Exception:
            home_final = None
            away_final = None
        sample = g.iloc[0]
        return pd.Series({
            "match_key": g.name,
            "soccer_match_id": sample.get("soccerMatchId"),
            "item_event_id": sample.get("itemEventId"),
            "item_event_date": sample.get("itemEventDate"),
            "team_home": sample.get("teamHome"),
            "team_away": sample.get("teamAway"),
            "team_home_id": sample.get("teamHomeId"),
            "team_away_id": sample.get("teamAwayId"),
            "home_score": home_final,
            "away_score": away_final
        })

    grouped = mss.groupby("match_key", group_keys=False).apply(final_for_group, include_groups=False)
    matches_df = grouped.reset_index(drop=True)

    if "soccer_match_id" in matches_df.columns:
        matches_df["soccer_match_id"] = pd.to_numeric(matches_df["soccer_match_id"], errors="coerce").astype("Int64")

    return matches_df

# -------------------------
# Final player_stats builder (includes exclude_from_statistics and matches_played)
# -------------------------
def build_player_stats(resp):
    players_df = build_players_df(resp)
    agg_stats = build_stats_from_matchstatistics(resp)
    matches_played = build_matches_played_from_itemplayerhistory(resp)

    # start from players to keep everyone (including those with zero stats)
    df = players_df[["player_id","name","exclude_from_statistics"]].copy()

    # left join aggregated stats
    if not agg_stats.empty:
        df = df.merge(agg_stats, on="player_id", how="left")
    else:
        df["goals"] = 0
        df["assists"] = 0
        df["own_goals"] = 0

    # ensure numeric existence
    for c in ("goals","assists","own_goals"):
        if c not in df.columns:
            df[c] = 0
    df[["goals","assists","own_goals"]] = df[["goals","assists","own_goals"]].fillna(0).astype(int)

    # merge matches_played
    if not matches_played.empty:
        df = df.merge(matches_played, on="player_id", how="left")
    else:
        df["matches_played"] = 0

    df["matches_played"] = df["matches_played"].fillna(0).astype(int)

    # sort and return
    df = df.sort_values(by=["goals","assists"], ascending=[False, False]).reset_index(drop=True)
    return df

# -------------------------
def main():
    assert INPUT_PATH.exists(), f"Input file not found: {INPUT_PATH}"
    top = load_json(INPUT_PATH)
    resp = top.get("response", {})

    # player stats (including exclude flag and matches_played)
    player_stats = build_player_stats(resp)
    print("\n=== player_stats (top 10) ===")
    if not player_stats.empty:
        with pd.option_context('display.max_columns', None, 'display.width', 200):
            print(player_stats.head(10).to_string(index=False))
    else:
        print("<no player stats available>")

    # matches table (one row per match)
    matches_df = build_matches_df(resp)
    print("\n=== matches_df (top 10) ===")
    if not matches_df.empty:
        with pd.option_context('display.max_columns', None, 'display.width', 200):
            print(matches_df.head(10).to_string(index=False))
    else:
        print("<no matches available>")

    # return structures if caller wants to use them programmatically
    return {"player_stats": player_stats, "matches_df": matches_df}

if __name__ == "__main__":
    main()
