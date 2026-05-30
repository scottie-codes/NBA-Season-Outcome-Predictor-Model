"""
make_master_features.py
Combines raw data files into a single feature table for model training.
Creates target variables (win percentage, made playoffs) from the data.
"""

import pandas as pd
from pathlib import Path

# File paths
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def normalize_columns(df):
    """
    Standardizes column names across the different API endpoints.
    The nba_api returns slightly different column names depending on the endpoint.
    """
    df = df.copy()

    # Standardize TEAM_NAME
    if 'TEAM' in df.columns and 'TEAM_NAME' not in df.columns:
        df.rename(columns={'TEAM': 'TEAM_NAME'}, inplace=True)
    if 'TeamName' in df.columns and 'TEAM_NAME' not in df.columns:
        df.rename(columns={'TeamName': 'TEAM_NAME'}, inplace=True)

    # Standardize TEAM_ID (sometimes merges create _x and _y versions)
    if 'TEAM_ID' not in df.columns:
        for alt in ['TEAM_ID_x', 'TEAM_ID_y', 'TeamID', 'teamId', 'TEAMID', 'team_id']:
            if alt in df.columns:
                df.rename(columns={alt: 'TEAM_ID'}, inplace=True)
                break

    # Standardize Season
    if 'Season' not in df.columns:
        for alt in ['SEASON', 'season', 'SeasonID', 'SEASON_ID']:
            if alt in df.columns:
                df.rename(columns={alt: 'Season'}, inplace=True)
                break

    return df


def load_raw_data():
    """Loads the three raw CSV files from the data collection phase."""
    partial = pd.read_csv(RAW / "nba_team_partial_game_logs.csv")
    reg = pd.read_csv(RAW / "nba_team_data_regular.csv")
    playoff = pd.read_csv(RAW / "nba_team_data_playoff.csv")

    return normalize_columns(partial), normalize_columns(reg), normalize_columns(playoff)


def get_win_percentage(reg):
    """
    Extract or calculate win percentage from regular season data.
    Endpoints are all different so will try a few different ways to get win %.
    This becomes our target variable for random forests regression model.
    """
    reg = reg.copy()

    # check if win percentage column already exists
    win_pct_names = ["W_PCT", "WIN_PCT", "WIN_PCT_BASE", "WINS_PCT", "WINPCT"]
    for col in reg.columns:
        if col.upper() in [name.upper() for name in win_pct_names]:
            reg["TARGET_WIN_PCT"] = pd.to_numeric(reg[col], errors="coerce")
            result = reg[["TEAM_ID", "Season", "TARGET_WIN_PCT"]].dropna().drop_duplicates()
            if not result.empty:
                return result

    # If not, calculate from wins and losses
    win_cols = ["W", "WINS", "WINS_BASE", "W_TOTAL"]
    loss_cols = ["L", "LOSSES", "LOSSES_BASE", "L_TOTAL"]

    w_col = next((c for c in win_cols if c in reg.columns), None)
    l_col = next((c for c in loss_cols if c in reg.columns), None)

    if w_col and l_col:
        wins = pd.to_numeric(reg[w_col], errors="coerce")
        losses = pd.to_numeric(reg[l_col], errors="coerce")
        reg["TARGET_WIN_PCT"] = (wins / (wins + losses)).astype(float)
        result = reg[["TEAM_ID", "Season", "TARGET_WIN_PCT"]].dropna().drop_duplicates()
        if not result.empty:
            return result

    # if that doesn't work, calculate from wins and games played
    gp_cols = ["GP", "GAMES_PLAYED", "G"]
    gp_col = next((c for c in gp_cols if c in reg.columns), None)

    if w_col and gp_col:
        wins = pd.to_numeric(reg[w_col], errors="coerce")
        games = pd.to_numeric(reg[gp_col], errors="coerce")
        reg["TARGET_WIN_PCT"] = (wins / games).astype(float)
        result = reg[["TEAM_ID", "Season", "TARGET_WIN_PCT"]].dropna().drop_duplicates()
        if not result.empty:
            return result

    raise ValueError("Could not find or calculate win percentage from data")


def get_playoff_labels(playoff_df):
    """
    Create binary playoff related labels:
    MADE_PLAYOFFS: 1 if team appears in playoff data, else 0
    ADV_ROUND1: 1 if team won at least 4 playoff games (won a series)
    """
    # Any team in playoff data made the playoffs
    base = playoff_df[["TEAM_ID", "Season"]].drop_duplicates().copy()
    base["MADE_PLAYOFFS"] = 1

    # check if they won a series (4 ormore wins)
    if "W" in playoff_df.columns:
        wins_by_team = playoff_df.groupby(["TEAM_ID", "Season"], as_index=False)["W"].sum()
        wins_by_team["ADV_ROUND1"] = (wins_by_team["W"] >= 4).astype(int)

        result = base.merge(
            wins_by_team[["TEAM_ID", "Season", "ADV_ROUND1"]],
            on=["TEAM_ID", "Season"],
            how="left"
        ).fillna({"ADV_ROUND1": 0})

        return result[["TEAM_ID", "Season", "MADE_PLAYOFFS", "ADV_ROUND1"]]

    # If no wins column, just return made_playoffs
    base["ADV_ROUND1"] = 0
    return base[["TEAM_ID", "Season", "MADE_PLAYOFFS", "ADV_ROUND1"]]


def build_master_features():
    """
    Main function to create the master features file.
    Combines partial game logs with target variables.
    """
    PROC.mkdir(parents=True, exist_ok=True)

    partial, reg, playoff = load_raw_data()

    win_pct = get_win_percentage(reg)
    playoff_labels = get_playoff_labels(playoff)

    # Merge features with targets
    df = partial.merge(win_pct, on=["TEAM_ID", "Season"], how="inner")
    df = df.merge(playoff_labels, on=["TEAM_ID", "Season"], how="left")
    df = df.fillna({"MADE_PLAYOFFS": 0, "ADV_ROUND1": 0})

    # Select feature columns
    cols_to_drop = {
        "TEAM_ID", "TEAM_NAME", "FullTeamName", "Season", "GamesAggregated",
        "TARGET_WIN_PCT", "ADV_ROUND1", "MADE_PLAYOFFS"
    }
    rank_cols = [c for c in df.columns if c.endswith("_RANK") or c == "PLUS_MINUS"]

    feature_cols = [c for c in df.columns if c not in cols_to_drop and c not in rank_cols]
    numeric_cols = df[feature_cols].select_dtypes("number").columns.tolist()

    # Build dataframe with IDs, targets, and features
    master = pd.concat([
        df[["TEAM_ID", "Season", "TARGET_WIN_PCT", "MADE_PLAYOFFS", "ADV_ROUND1"]],
        df[numeric_cols]
    ], axis=1)

    out_path = PROC / "master_features.csv"
    master.to_csv(out_path, index=False)

    print(f"Created master_features.csv: {master.shape[0]} rows, {master.shape[1]} columns")
    print(f"Saved to {out_path}")
    print(f"Playoff teams: {int(master['MADE_PLAYOFFS'].sum())}")

    return out_path


if __name__ == "__main__":
    build_master_features()