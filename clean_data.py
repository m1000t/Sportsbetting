import pandas as pd
import os

def american_to_decimal(odds):
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def implied_probability(decimal_odds):
    return 1 / decimal_odds

def add_rolling_features(df):
    df = df.sort_values("date").copy()

    # Create win flags for home and away teams
    df["home_win_flag"] = (df["score_home"] > df["score_away"]).astype(int)
    df["away_win_flag"] = (df["score_away"] > df["score_home"]).astype(int)

    df["home_last5_winrate"] = 0.5  # default neutral
    df["away_last5_winrate"] = 0.5

    # Prepare team games dictionary
    teams = pd.unique(pd.concat([df["home"], df["away"]]))



    team_games = {}
    for team in teams:
        home_games = df[df["home"] == team][["date", "home_win_flag"]].rename(columns={"home_win_flag": "win"})
        away_games = df[df["away"] == team][["date", "away_win_flag"]].rename(columns={"away_win_flag": "win"})
        all_games = pd.concat([home_games, away_games]).sort_values("date")
        team_games[team] = all_games

    def rolling_win_rate(team_games_df, date):
        past_games = team_games_df[team_games_df["date"] < date].tail(5)
        if len(past_games) == 0:
            return 0.5
        return past_games["win"].mean()

    home_wr = []
    away_wr = []

    for idx, row in df.iterrows():
        home_wr.append(rolling_win_rate(team_games[row["home"]], row["date"]))
        away_wr.append(rolling_win_rate(team_games[row["away"]], row["date"]))

    df["home_last5_winrate"] = home_wr
    df["away_last5_winrate"] = away_wr

    return df

def preprocess_data(path_in="nba_2008-2025.csv",
                    path_out="cleaned_data.csv"):
    df = pd.read_csv(path_in)
    print(f"Loaded raw data: {df.shape[0]} rows")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df.dropna(subset=["moneyline_home", "moneyline_away", "score_home", "score_away", "spread", "total"])

    df["home_decimal"] = df["moneyline_home"].apply(american_to_decimal)
    df["away_decimal"] = df["moneyline_away"].apply(american_to_decimal)

    df["home_implied_prob"] = df["home_decimal"].apply(implied_probability)
    df["away_implied_prob"] = df["away_decimal"].apply(implied_probability)

    df["home_win"] = (df["score_home"] > df["score_away"]).astype(int)

    # Add rolling features
    df = add_rolling_features(df)

    keep_cols = [
        "date", "home", "away",
        "score_home", "score_away",
        "moneyline_home", "moneyline_away",
        "home_decimal", "away_decimal",
        "home_implied_prob", "away_implied_prob",
        "spread", "total",
        "home_last5_winrate", "away_last5_winrate",
        "home_win"
    ]
    df_cleaned = df[keep_cols]

    df_cleaned.to_csv(path_out, index=False)
    print(f"Saved cleaned data with rolling features: {df_cleaned.shape[0]} rows → {path_out}")

    return df_cleaned

if __name__ == "__main__":
    preprocess_data()
