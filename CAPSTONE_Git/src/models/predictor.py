import numpy as np
import pandas as pd
import joblib

from src.models.data_io import (
    load_master, list_seasons, list_teams_for_season, get_team_row,
    artifact_paths, load_feature_list
)

def load_models():
    clf_pkl, reg_pkl, feat_json = artifact_paths()
    clf = joblib.load(clf_pkl)
    reg = joblib.load(reg_pkl)
    feats = load_feature_list()
    return clf, reg, feats

def prepare_features(row, feat_cols):
    x = []
    for c in feat_cols:
        val = row[c] if c in row.index else 0.0
        if pd.isna(val): val = 0.0
        x.append(float(val))
    return np.array(x, dtype=np.float32).reshape(1, -1)

def predict_for(season, team_id):
    df = load_master()
    row = get_team_row(season, team_id, df)
    clf, reg, feats = load_models()
    X = prepare_features(row, feats)

    out = {
        "season": season,
        "team_id": int(team_id),
        "team_name": None
    }
    if "TEAM_NAME" in row.index and pd.notna(row["TEAM_NAME"]):
        out["team_name"] = str(row["TEAM_NAME"])
    elif "FullTeamName" in row.index and pd.notna(row["FullTeamName"]):
        out["team_name"] = str(row["FullTeamName"])

    # classifier
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        out["prob_top"] = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
        out["pred_top"] = int(out["prob_top"] >= 0.5)
    else:
        pred = clf.predict(X)
        out["pred_top"] = int(pred[0])
        out["prob_top"] = None

    # regression
    if hasattr(reg, "predict"):
        winp = reg.predict(X)
        out["win_pct_pred"] = float(winp[0])
    else:
        out["win_pct_pred"] = None

    # actuals
    if "TARGET_WIN_PCT" in row.index:
        out["win_pct_actual"] = float(row["TARGET_WIN_PCT"])
    if "WINPCT_TOP" in row.index:
        out["top_actual"] = int(row["WINPCT_TOP"])

    return out

def get_seasons():
    return list_seasons()

def get_teams(season):
    return list_teams_for_season(season)