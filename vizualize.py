import matplotlib.pyplot as plt
import numpy as np

def plot_threshold_vs_winrate(model, X_test, y_test, df_test):
    thresholds = np.arange(0.5, 0.9, 0.01)
    win_rates = []

    for thresh in thresholds:
        preds = model.predict_proba(X_test)[:, 1]
        bet_indices = preds > thresh
        bets = y_test[bet_indices]

        if len(bets) > 0:
            win_rate = (bets == 1).mean() * 100
            win_rates.append(win_rate)
        else:
            win_rates.append(np.nan)

    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, win_rates, marker='o')
    plt.title("Win Rate vs. Confidence Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Win Rate (%)")
    plt.grid(True)
    plt.show()


def plot_bankroll(model, X_test, y_test, df_test, threshold=0.55, bet_amount=10):
    preds = model.predict_proba(X_test)[:, 1]
    bankroll = [100]  # starting bankroll
    current = 100

    for i, prob in enumerate(preds):
        if prob > threshold:
            row = df_test.iloc[i]
            if y_test.iloc[i] == 1:
                odds = 1 / row["home_implied_prob"]
                payout = bet_amount * odds
                current += payout - bet_amount
            else:
                current -= bet_amount
        bankroll.append(current)

    plt.figure(figsize=(10, 5))
    plt.plot(bankroll)
    plt.title("Cumulative Bankroll Over Time")
    plt.xlabel("Bet Number")
    plt.ylabel("Bankroll ($)")
    plt.grid(True)
    plt.show()
