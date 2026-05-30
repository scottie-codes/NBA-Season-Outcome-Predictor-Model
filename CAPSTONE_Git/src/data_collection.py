"""
data_collection.py
Pulls and parses NBA team data from the nba_api library and saves to CSV files.
Collects regular season stats, playoff stats, and first games of each season.
Models don't use all data, but thorough, complete data was taken for creation of future models.
"""

import pandas as pd
import os
import time
from nba_api.stats.endpoints import leaguestandings, leaguedashteamstats, teamgamelogs


def get_team_standings(season):
    """Get standings (wins, losses and rankings) for a season."""
    for attempt in range(3):
        try:
            data = leaguestandings.LeagueStandings(season=season)
            df = data.get_data_frames()[0]

            if 'TeamID' in df.columns:
                df.rename(columns={'TeamID': 'TEAM_ID'}, inplace=True)
            if 'TeamName' in df.columns:
                df.rename(columns={'TeamName': 'TEAM_NAME'}, inplace=True)

            df['Season'] = season
            df['FullTeamName'] = df['TeamCity'] + ' ' + df['TEAM_NAME']
            return df
        except Exception as e:
            print(f"Standings error for {season}: {str(e)[:60]}")
            time.sleep(4)

    return pd.DataFrame()


def get_league_base_stats(season, season_type="Regular Season"):
    """Get basic stats (points, rebounds, assists, etc.) for all teams."""
    for attempt in range(3):
        try:
            time.sleep(2)  # Pause to avoid API rate limits

            data = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                per_mode_detailed='PerGame',
                season_type_all_star=season_type,
                measure_type_detailed_defense='Base'
            )
            df = data.get_data_frames()[0]

            if 'TEAM_NAME' not in df.columns and 'TEAM' in df.columns:
                df.rename(columns={'TEAM': 'TEAM_NAME'}, inplace=True)

            df['Season'] = season
            df['FullTeamName'] = df['TEAM_NAME']
            return df
        except Exception as e:
            print(f"Base stats error for {season}: {str(e)[:60]}")
            time.sleep(4)

    return pd.DataFrame()


def get_league_advanced_stats(season, season_type="Regular Season"):
    """Gets advanced stats (offensive rating, pace, etc.)."""
    for attempt in range(3):
        try:
            time.sleep(2)

            data = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                per_mode_detailed='PerGame',
                season_type_all_star=season_type,
                measure_type_detailed_defense='Advanced'
            )
            df = data.get_data_frames()[0]

            if 'TEAM_NAME' not in df.columns and 'TEAM' in df.columns:
                df.rename(columns={'TEAM': 'TEAM_NAME'}, inplace=True)

            df['Season'] = season
            df['FullTeamName'] = df['TEAM_NAME']
            return df
        except Exception as e:
            print(f"Advanced stats error for {season}: {str(e)[:60]}")
            time.sleep(4)

    return pd.DataFrame()


def combine_regular_season(start_year=2015, end_year=2024):
    """Combines regular season data for multiple years into one dataframe."""
    all_seasons = []

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"  # Format: "2015-16"
        print(f"Processing {season}...")

        standings_df = get_team_standings(season)
        if standings_df.empty:
            continue

        time.sleep(2)

        base_df = get_league_base_stats(season, season_type="Regular Season")
        if base_df.empty:
            continue

        adv_df = get_league_advanced_stats(season, season_type="Regular Season")
        if adv_df.empty:
            continue

        merged_stats = pd.merge(
            base_df,
            adv_df,
            on=['TEAM_ID', 'TEAM_NAME', 'Season', 'FullTeamName'],
            how='inner',
            suffixes=('_base', '_adv')
        )

        full = pd.merge(
            standings_df,
            merged_stats,
            on=['Season', 'FullTeamName'],
            how='inner'
        )
        all_seasons.append(full)
        time.sleep(3)

    if not all_seasons:
        return pd.DataFrame(), 0, 0

    combined = pd.concat(all_seasons, ignore_index=True)
    num_teams = combined['FullTeamName'].nunique()
    num_seasons = combined['Season'].nunique()
    expected_records = num_teams * num_seasons

    return combined, expected_records, combined.shape[0]


def combine_playoff_season(start_year=2015, end_year=2024):
    """Combine playoff data for multiple years into one dataframe."""
    all_seasons = []

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        print(f"Processing {season} playoffs...")

        base_df = get_league_base_stats(season, season_type="Playoffs")
        if base_df.empty:
            continue

        adv_df = get_league_advanced_stats(season, season_type="Playoffs")
        if adv_df.empty:
            continue

        merged_stats = pd.merge(
            base_df,
            adv_df,
            on=['TEAM_ID', 'TEAM_NAME', 'Season', 'FullTeamName'],
            how='inner',
            suffixes=('_base', '_adv')
        )
        all_seasons.append(merged_stats)
        time.sleep(3)

    if not all_seasons:
        return pd.DataFrame(), 0, 0

    combined = pd.concat(all_seasons, ignore_index=True)
    num_teams = combined['FullTeamName'].nunique()
    num_seasons = combined['Season'].nunique()
    expected_records = num_teams * num_seasons

    return combined, expected_records, combined.shape[0]


def get_teams_for_season(season):
    """Get list of teams that played in a given season."""
    standings_df = get_team_standings(season)
    if standings_df.empty:
        return pd.DataFrame()

    if 'TEAM_ID' not in standings_df.columns:
        print(f"TEAM_ID not found for {season}")
        return pd.DataFrame()

    return standings_df[['TEAM_ID', 'FullTeamName']].drop_duplicates()


def get_team_game_logs(team_id, season, season_type="Regular Season"):
    """Get game-by-game stats for a specific team and season."""
    for attempt in range(3):
        try:
            data = teamgamelogs.TeamGameLogs(
                team_id_nullable=team_id,
                season_nullable=season,
                season_type_nullable=season_type
            )
            df = data.get_data_frames()[0]

            if df.empty:
                time.sleep(2)
                continue

            df['Season'] = season
            df['TEAM_ID'] = team_id
            return df

        except Exception as e:
            print(f"Error for team {team_id}: {str(e)[:60]}")
            time.sleep(3)

    return pd.DataFrame()


def combine_partial_game_logs(start_year=2015, end_year=2024, num_games=20):
    """
    Get first N games of each season for each team.
    This is used for early season predictions.
    Will use 20 games, but can change.
    """
    all_seasons = []

    for year in range(start_year, end_year + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        print(f"Processing {season}...")

        teams = get_teams_for_season(season)
        if teams.empty:
            continue

        season_team_logs = []

        for idx, row in teams.iterrows():
            team_id = row['TEAM_ID']
            team_name = row['FullTeamName']

            team_logs = get_team_game_logs(team_id, season)
            if team_logs.empty:
                continue

            try:
                # Sort by date and take first N games
                team_logs['GAME_DATE'] = pd.to_datetime(team_logs['GAME_DATE'])
                team_logs = team_logs.sort_values('GAME_DATE')
                partial_logs = team_logs.head(num_games)

                # Get numeric columns for averaging
                numeric_cols = partial_logs.select_dtypes(include='number').columns.tolist()

                # Don't average ID columns
                for col in ['TEAM_ID', 'GAME_ID']:
                    if col in numeric_cols:
                        numeric_cols.remove(col)

                if not numeric_cols:
                    continue

                # Calculate averages for first N games
                agg_stats = partial_logs[numeric_cols].mean().to_frame().T
                agg_stats['TEAM_ID'] = team_id
                agg_stats['TEAM_NAME'] = team_name
                agg_stats['Season'] = season
                agg_stats['FullTeamName'] = team_name
                agg_stats['GamesAggregated'] = len(partial_logs)

                season_team_logs.append(agg_stats)

            except Exception as e:
                print(f"Error for {team_name}: {str(e)[:60]}")

            time.sleep(2)

        if season_team_logs:
            season_df = pd.concat(season_team_logs, ignore_index=True)
            all_seasons.append(season_df)

        time.sleep(3)

    if not all_seasons:
        return pd.DataFrame(), 0, 0

    combined = pd.concat(all_seasons, ignore_index=True)
    num_teams = combined['FullTeamName'].nunique()
    num_seasons = combined['Season'].nunique()
    expected_records = num_teams * num_seasons

    return combined, expected_records, combined.shape[0]


if __name__ == "__main__":
    start_year = 2015
    end_year = 2024
    num_games_to_aggregate = 20

    # Collect first 20 games per team per season
    print(f"Collecting first {num_games_to_aggregate} games per team\n")

    df_logs, expected_logs, actual_logs = combine_partial_game_logs(
        start_year, end_year, num_games=num_games_to_aggregate
    )

    if not df_logs.empty:
        os.makedirs('../data/raw', exist_ok=True)
        out_logs = '../data/raw/nba_team_partial_game_logs.csv'
        df_logs.to_csv(out_logs, index=False)
        print(f"\nGame logs: {df_logs.shape[0]} records, {df_logs['Season'].nunique()} seasons")

    # Collect regular season data
    print("\nCollecting regular season stats\n")

    df_reg, expected_reg, actual_reg = combine_regular_season(start_year, end_year)

    if not df_reg.empty:
        os.makedirs('../data/raw', exist_ok=True)
        out_reg = '../data/raw/nba_team_data_regular.csv'
        df_reg.to_csv(out_reg, index=False)
        print(f"\nRegular season: {df_reg.shape[0]} records")

    # Collect playoff data
    print("\nCollecting playoff stats\n")

    df_playoff, expected_playoff, actual_playoff = combine_playoff_season(start_year, end_year)

    if not df_playoff.empty:
        os.makedirs('../data/raw', exist_ok=True)
        out_playoff = '../data/raw/nba_team_data_playoff.csv'
        df_playoff.to_csv(out_playoff, index=False)
        print(f"\nPlayoff: {df_playoff.shape[0]} records")

    print(f"\nTotal: {actual_reg} regular, {actual_playoff} playoff, {actual_logs} game logs")