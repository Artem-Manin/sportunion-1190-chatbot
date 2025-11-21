# src/build_tables.py
import json
from pathlib import Path
from typing import Dict, Any

import requests
import pandas as pd

# -------------------------------------------------
# Season config: URLs + local JSON targets
# -------------------------------------------------
SEASONS = {
    "25/26": {
        "url": "https://api.ekatime.com/api/v001/soccer/public-statistics/sportunion-1190-2526",
        "file": Path("data/season_2526.json"),
    },
    "24/25": {
        "url": "https://api.ekatime.com/api/v001/soccer/public-statistics/sportunion-1190-2425",
        "file": Path("data/season_2425.json"),
    },
}

# Ensure data dir exists
Path("data").mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# Generic helpers
# -------------------------------------------------
def _load_local_json(file_path: Path) -> Dict[str, Any] | None:
    """Load JSON from a local file if it exists, else return None."""
    if not file_path.exists():
        print(f"❌ Local file not found: {file_path}")
        return None
    print(f"➡ Using local file {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def download_season_json(
    season: str,
    url: str,
    file_path: Path,
    allow_download: bool,
) -> Dict[str, Any] | None:
    """
    Behavior:
    - If allow_download is False:
        -> Only load from local file_path (no HTTP).
    - If allow_download is True:
        -> Try HTTP download, save to file_path.
           On failure, fallback to local file_path if present.

    Returns parsed JSON dict or None.
    """
    print(f"\n=== Season {season} ===")

    if not allow_download:
        print(f"Skipping download for season {season}, using local file only.")
        return _load_local_json(file_path)

    print(f"Downloading from {url} ...")
    data = None

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # save downloaded JSON
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"✔ Downloaded and saved to {file_path}")
    except Exception as e:
        print(f"⚠ Download failed for season {season}: {e}")
        # fallback to existing local
        data = _load_local_json(file_path)

    return data


def to_df(obj):
    """Convert list/dict to pandas DataFrame, flattening nested dicts."""
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, list):
        if len(obj) == 0:
            return pd.DataFrame()
        if isinstance(obj[0], dict):
            return pd.json_normalize(obj, sep=".")
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        return pd.json_normalize([obj], sep=".")
    return pd.DataFrame([obj])


# -------------------------------------------------
# Players + stats builders
# -------------------------------------------------
def build_players_df(resp: Dict[str, Any]) -> pd.DataFrame:
    players = resp.get("players", [])
    df = to_df(players)

    mapping = {
        "baseObjectId": "player_id",
        "name": "name",
        "excludeFromStatistics": "exclude_from_statistics",
        "createdTimestamp": "created_ts",
        "updatedTimestamp": "updated_ts",
    }
    present = [c for c in mapping if c in df.columns]
    if present:
        players_df = df[present].rename(columns=mapping)
    else:
        players_df = pd.DataFrame(columns=list(mapping.values()))

    # types & defaults
    players_df["player_id"] = pd.to_numeric(players_df.get("player_id", pd.NA), errors="coerce").astype("Int64")
    players_df["name"] = players_df.get("name", "").fillna("").astype(str)
    players_df["exclude_from_statistics"] = (
        players_df.get("exclude_from_statistics", False).fillna(False).astype(bool)
    )

    if "created_ts" in players_df.columns:
        players_df["created_ts"] = pd.to_numeric(players_df["created_ts"], errors="coerce").astype("Int64")
    if "updated_ts" in players_df.columns:
        players_df["updated_ts"] = pd.to_numeric(players_df["updated_ts"], errors="coerce").astype("Int64")

    return players_df.reset_index(drop=True)


def build_stats_from_matchstatistics(resp: Dict[str, Any]) -> pd.DataFrame:
    ms = to_df(resp.get("matchStatistics", []))
    if ms.empty:
        return pd.DataFrame(columns=["player_id", "goals", "assists", "own_goals"])

    player_col = next((c for c in ("playerId", "player_id", "playerID") if c in ms.columns), None)
    if player_col is None:
        return pd.DataFrame(columns=["player_id", "goals", "assists", "own_goals"])

    ms[player_col] = pd.to_numeric(ms[player_col], errors="coerce").astype("Int64")

    def pick_flag(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
        for n in names:
            if n in df.columns:
                return pd.to_numeric(df[n], errors="coerce").fillna(0).astype(int)
        return pd.Series(0, index=df.index, dtype=int)

    goals = pick_flag(ms, ("score", "isGoal", "Score"))
    assists = pick_flag(ms, ("assist", "isAssist", "Assist"))
    own_goals = pick_flag(ms, ("ownGoal", "own_goal", "OwnGoal"))

    ms["_pid"] = ms[player_col]

    agg = (
        ms.groupby("_pid", as_index=False)
        .agg(
            goals_sum=("score", lambda s: int(goals.loc[s.index].sum())),
            assists_sum=("score", lambda s: int(assists.loc[s.index].sum())),
            own_goals_sum=("score", lambda s: int(own_goals.loc[s.index].sum())),
        )
        .rename(columns={"_pid": "player_id"})
    )

    agg["player_id"] = pd.to_numeric(agg["player_id"], errors="coerce").astype("Int64")
    agg = agg.rename(columns={"goals_sum": "goals", "assists_sum": "assists", "own_goals_sum": "own_goals"})
    return agg[["player_id", "goals", "assists", "own_goals"]]


def build_matches_played_from_itemplayerhistory(resp: Dict[str, Any]) -> pd.DataFrame:
    iph = to_df(resp.get("itemPlayerHistory", []))
    if iph.empty:
        return pd.DataFrame(columns=["player_id", "matches_played"])

    player_col = next((c for c in ("playerId", "player_id", "playerID") if c in iph.columns), None)
    if player_col is None:
        return pd.DataFrame(columns=["player_id", "matches_played"])

    iph["player_id"] = pd.to_numeric(iph[player_col], errors="coerce").astype("Int64")

    if "soccerMatchId" in iph.columns:
        iph["soccer_match_id"] = pd.to_numeric(iph["soccerMatchId"], errors="coerce").astype("Int64")
    else:
        iph["soccer_match_id"] = pd.NA

    iph["item_event_id_str"] = iph.get("itemEventId", "").astype(str).fillna("")

    def choose_match_id(row):
        sm = row["soccer_match_id"]
        try:
            if pd.notna(sm) and int(sm) != 0:
                return f"s{int(sm)}"
        except Exception:
            pass
        return f"ie{row['item_event_id_str']}"

    iph["match_identifier"] = iph.apply(choose_match_id, axis=1)

    mp = (
        iph.groupby("player_id", as_index=False)["match_identifier"]
        .nunique()
        .rename(columns={"match_identifier": "matches_played"})
    )
    mp["matches_played"] = mp["matches_played"].astype(int)
    return mp


# -------------------------------------------------
# Matches builder
# -------------------------------------------------
def build_matches_df(resp: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a one-row-per-match table from matchScoreStatistics,
    using groupby().agg(...) instead of deprecated groupby.apply.
    """
    mss = to_df(resp.get("matchScoreStatistics", []))
    if mss.empty:
        return pd.DataFrame()

    needed_cols = [
        "soccerMatchId",
        "itemEventId",
        "itemEventDate",
        "teamHome",
        "teamAway",
        "teamHomeId",
        "teamAwayId",
        "scoreTeamHome",
        "scoreTeamAway",
    ]
    for c in needed_cols:
        if c not in mss.columns:
            mss[c] = pd.NA

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

    mss["scoreTeamHome_num"] = pd.to_numeric(mss["scoreTeamHome"], errors="coerce")
    mss["scoreTeamAway_num"] = pd.to_numeric(mss["scoreTeamAway"], errors="coerce")

    grouped = (
        mss.sort_values("itemEventDate")
        .groupby("match_key", as_index=False)
        .agg(
            soccer_match_id=("soccerMatchId", "first"),
            item_event_id=("itemEventId", "first"),
            item_event_date=("itemEventDate", "first"),
            team_home=("teamHome", "first"),
            team_away=("teamAway", "first"),
            team_home_id=("teamHomeId", "first"),
            team_away_id=("teamAwayId", "first"),
            home_score=("scoreTeamHome_num", "max"),
            away_score=("scoreTeamAway_num", "max"),
        )
    )

    grouped["soccer_match_id"] = pd.to_numeric(grouped["soccer_match_id"], errors="coerce").astype("Int64")
    grouped["home_score"] = grouped["home_score"].astype("Int64")
    grouped["away_score"] = grouped["away_score"].astype("Int64")

    return grouped


def build_player_stats(resp: Dict[str, Any]) -> pd.DataFrame:
    players_df = build_players_df(resp)
    agg_stats = build_stats_from_matchstatistics(resp)
    matches_played = build_matches_played_from_itemplayerhistory(resp)

    df = players_df[["player_id", "name", "exclude_from_statistics"]].copy()

    if not agg_stats.empty:
        df = df.merge(agg_stats, on="player_id", how="left")
    else:
        df["goals"] = 0
        df["assists"] = 0
        df["own_goals"] = 0

    for c in ("goals", "assists", "own_goals"):
        if c not in df.columns:
            df[c] = 0
    df[["goals", "assists", "own_goals"]] = df[["goals", "assists", "own_goals"]].fillna(0).astype(int)

    if not matches_played.empty:
        df = df.merge(matches_played, on="player_id", how="left")
    else:
        df["matches_played"] = 0
    df["matches_played"] = df["matches_played"].fillna(0).astype(int)

    df = df.sort_values(by=["goals", "assists"], ascending=[False, False]).reset_index(drop=True)
    return df


# -------------------------------------------------
# MAIN: process both seasons, add `season`, combine
# -------------------------------------------------
def main(download_2526: bool = False):
    """
    Build combined tables for all seasons.

    - Season 24/25 is ALWAYS loaded from local JSON (no HTTP).
    - Season 25/26:
        * If download_2526=True  -> try to download & update local file, fallback to local.
        * If download_2526=False -> use local file only.
    """
    all_player_stats = []
    all_matches = []

    for season, cfg in SEASONS.items():
        if season == "24/25":
            allow_download = False
        elif season == "25/26":
            allow_download = download_2526
        else:
            allow_download = False

        data = download_season_json(season, cfg["url"], cfg["file"], allow_download=allow_download)
        if data is None:
            continue

        resp = data.get("response", data)

        print(f"\n--- Building tables for season {season} ---")
        player_stats = build_player_stats(resp)
        matches_df = build_matches_df(resp)

        player_stats["season"] = season
        matches_df["season"] = season

        print("\nplayer_stats (top 10):")
        if not player_stats.empty:
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(player_stats.head(10).to_string(index=False))
        else:
            print("<no player stats>")

        print("\nmatches_df (top 10):")
        if not matches_df.empty:
            with pd.option_context("display.max_columns", None, "display.width", 200):
                print(matches_df.head(10).to_string(index=False))
        else:
            print("<no matches>")

        all_player_stats.append(player_stats)
        all_matches.append(matches_df)

    if all_player_stats:
        player_stats_all = pd.concat(all_player_stats, ignore_index=True)
    else:
        player_stats_all = pd.DataFrame()

    if all_matches:
        matches_all = pd.concat(all_matches, ignore_index=True)
    else:
        matches_all = pd.DataFrame()

    print("\n=== COMBINED player_stats_all (top 10) ===")
    if not player_stats_all.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(player_stats_all.head(10).to_string(index=False))
    else:
        print("<empty>")

    print("\n=== COMBINED matches_all (top 10) ===")
    if not matches_all.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(matches_all.head(10).to_string(index=False))
    else:
        print("<empty>")

    return {
        "player_stats_all": player_stats_all,
        "matches_all": matches_all,
    }


if __name__ == "__main__":
    # If you run this file directly, you can choose whether to download 25/26
    main(download_2526=True)
