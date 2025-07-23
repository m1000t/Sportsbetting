import pandas as pd

def backtest(model, X_test, y_test, df_test, threshold=0.02, bet_amount=40, starting_bankroll=50):
    bankroll = starting_bankroll
    bet_history = []

    # Get model predicted probability for home win
    probs = model.predict_proba(X_test)[:, 1]  # Probability of home_win=1

    for i in range(len(probs)):
        model_prob = probs[i]
        book_prob = df_test.iloc[i]["home_implied_prob"]
        decimal_odds = df_test.iloc[i]["home_decimal"]
        actual = y_test.iloc[i]

        # Bet only if model probability exceeds bookmaker implied probability by threshold
        if model_prob > book_prob + threshold:
            # Place bet on home team
            if actual == 1:
                profit = bet_amount * (decimal_odds - 1)
                bankroll += profit
                bet_result = "Win"
            else:
                bankroll -= bet_amount
                bet_result = "Loss"

            bet_history.append({
                "date": df_test.iloc[i]["date"],
                "home_team": df_test.iloc[i]["home"],
                "away_team": df_test.iloc[i]["away"],
                "model_prob": model_prob,
                "book_prob": book_prob,
                "bet_amount": bet_amount,
                "result": bet_result,
                "bankroll": bankroll
            })

    total_bets = len(bet_history)
    wins = sum(1 for bet in bet_history if bet["result"] == "Win")
    roi = (bankroll - starting_bankroll) / starting_bankroll * 100
    win_rate = wins / total_bets if total_bets > 0 else 0

    print("\nBacktest results:")
    print(f"Total bets placed: {total_bets}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Final bankroll: ${bankroll:.2f}")
    print(f"ROI: {roi:.2f}%")

    # Optional: save bet history to CSV
    pd.DataFrame(bet_history).to_csv("bet_history.csv", index=False)

    return bankroll, bet_history

# Example usage inside your train script:
if __name__ == "__main__":
    from model import train_improved_model  # or wherever your train function is
    model, X_test, y_test, df_test = train_improved_model()
    backtest(model, X_test, y_test, df_test)
