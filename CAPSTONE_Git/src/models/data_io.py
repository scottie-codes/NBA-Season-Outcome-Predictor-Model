from pathlib import Path
import json
import pandas as pd

# Go from src/models/ to src to project root
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROC = DATA / "processed"
RAW  = DATA / "raw"

MF           = PROC / "master_features.csv"
MF_CLUSTERS  = PROC / "master_features_with_clusters.csv"
METRICS_CLS  = PROC / "metrics_classification.csv"
METRICS_REG  = PROC / "metrics_regression.csv"
PRED_TEST    = PROC / "predictions_test.csv"

CLF_PKL      = PROC / "classifier.pkl"
REG_PKL      = PROC / "regressor.pkl"
CLF_FEATURES = PROC / "clf_features.json"


def _find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cols.get(cand.lower())
        if c is not None:
            return c
    return None

def _compose_team_name(df):
    c_city = _find_col(df, ["TeamCity","CITY"])
    c_team = _find_col(df, ["TEAM_NAME","TeamName","TEAM","TEAMNAME"])
    if c_team and c_city:
        return df[c_city].astype(str) + " " + df[c_team].astype(str)
    if c_team:
        return df[c_team].astype(str)
    return None

def _load_names_from_raw():
    """Builds a lookup table to get team names from raw Csv"""
    candidates = [
        RAW / "nba_team_data_regular.csv",
        RAW / "nba_team_data_playoff.csv",
        RAW / "nba_team_partial_game_logs.csv",
    ]
    parts = []
    for p in candidates:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            if "TEAM_ID" not in df.columns or "Season" not in df.columns:
                continue
            name_series = _compose_team_name(df)
            if name_series is None:
                c_full = _find_col(df, ["FullTeamName"])
                if c_full:
                    name_series = df[c_full].astype(str)
            if name_series is None:
                name_series = df["TEAM_ID"].astype(str)
            part = pd.DataFrame({
                "Season": df["Season"].astype(str),
                "TEAM_ID": df["TEAM_ID"].astype(int),
                "TEAM_NAME": name_series.astype(str),
            })
            parts.append(part)
        except Exception:
            pass
    if not parts:
        return pd.DataFrame(columns=["Season","TEAM_ID","TEAM_NAME"])
    cat = pd.concat(parts, ignore_index=True)
    cat["name_len"] = cat["TEAM_NAME"].str.len()
    cat = cat.sort_values(["Season","TEAM_ID","name_len"], ascending=[True, True, False])
    cat = cat.drop_duplicates(subset=["Season","TEAM_ID"])[["Season","TEAM_ID","TEAM_NAME"]]
    return cat

def _names_catalog():
    """Combine names from clusters and raw into single catalog."""
    pieces = []
    if MF_CLUSTERS.exists():
        c = pd.read_csv(MF_CLUSTERS)
        if "Season" in c.columns and "TEAM_ID" in c.columns:
            name_col = _find_col(c, ["TEAM_NAME","FullTeamName","TeamName","TEAM","TEAMNAME"])
            if name_col:
                cc = c[["Season","TEAM_ID",name_col]].dropna()
                cc = cc.rename(columns={name_col:"TEAM_NAME"})
                pieces.append(cc.astype({"TEAM_ID":"int"}).assign(Season=lambda d: d["Season"].astype(str)))
    raw_cat = _load_names_from_raw()
    if not raw_cat.empty:
        pieces.append(raw_cat)
    if not pieces:
        return pd.DataFrame(columns=["Season","TEAM_ID","TEAM_NAME"])
    cat = pd.concat(pieces, ignore_index=True)
    cat = cat.dropna(subset=["Season","TEAM_ID"])
    cat["name_len"] = cat["TEAM_NAME"].str.len()
    cat = cat.sort_values(["Season","TEAM_ID","name_len"], ascending=[True, True, False])
    cat = cat.drop_duplicates(subset=["Season","TEAM_ID"])[["Season","TEAM_ID","TEAM_NAME"]]
    return cat

def _ensure_team_name(df):
    df = df.copy()
    name_col = _find_col(df, ["TEAM_NAME","FullTeamName","TeamName","TEAM","TEAMNAME"])
    if name_col:
        if name_col != "TEAM_NAME":
            df["TEAM_NAME"] = df[name_col]
        return df

    comp = _compose_team_name(df)
    if comp is not None:
        df["TEAM_NAME"] = comp
        return df

    if "Season" in df.columns and "TEAM_ID" in df.columns:
        cat = _names_catalog()
        if not cat.empty:
            merged = df.merge(cat, on=["Season","TEAM_ID"], how="left")
            if "TEAM_NAME" in merged.columns:
                merged["TEAM_NAME"] = merged["TEAM_NAME"].fillna(merged["TEAM_ID"].astype(str))
                return merged

    if "TEAM_ID" in df.columns:
        df["TEAM_NAME"] = df["TEAM_ID"].astype(str)
    else:
        df["TEAM_NAME"] = "Unknown"
    return df


def load_master():
    if not MF.exists():
        raise FileNotFoundError(f"Missing file: {MF}")
    df = pd.read_csv(MF)
    return _ensure_team_name(df)

def load_master_with_clusters():
    if MF_CLUSTERS.exists():
        df = pd.read_csv(MF_CLUSTERS)
    else:
        df = load_master()
    return _ensure_team_name(df)

def list_seasons(df=None):
    if df is None: df = load_master()
    return sorted(df["Season"].astype(str).dropna().unique().tolist())

def list_teams_for_season(season, df=None):
    if df is None: df = load_master()
    sdf = df[df["Season"].astype(str) == str(season)].copy()
    out = sdf[["TEAM_ID","TEAM_NAME"]].drop_duplicates()
    return out.sort_values(["TEAM_NAME","TEAM_ID"])

def get_team_row(season, team_id, df=None):
    if df is None: df = load_master()
    m = (df["Season"].astype(str) == str(season)) & (df["TEAM_ID"].astype(int) == int(team_id))
    row = df.loc[m]
    if not row.empty:
        return row.iloc[0]
    c = load_master_with_clusters()
    m2 = (c["Season"].astype(str) == str(season)) & (c["TEAM_ID"].astype(int) == int(team_id))
    crow = c.loc[m2]
    if not crow.empty:
        return crow.iloc[0]
    raise ValueError(f"No row for season={season}, team_id={team_id}")

def artifact_paths():
    return CLF_PKL, REG_PKL, CLF_FEATURES

def load_feature_list():
    if not CLF_FEATURES.exists():
        df = load_master()
        drop = {"TEAM_ID","Season","TARGET_WIN_PCT","WINPCT_TOP","ADV_ROUND1","ADV_ROUND2","ADV_ROUND3","ADV_FINALS"}
        feats = [c for c in df.select_dtypes("number").columns if c not in drop]
        return feats
    with open(CLF_FEATURES, "r") as f:
        return json.load(f)