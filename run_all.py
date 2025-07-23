from clean_data import preprocess_data
from model import train_improved_model
from backtest import backtest
from vizualize import plot_bankroll

def main():
    preprocess_data()
    model, X_test, y_test, df_test = train_improved_model()
    backtest(model, X_test, y_test, df_test, threshold=0.02, bet_amount=40, starting_bankroll=50)
    plot_bankroll(model, X_test, y_test, df_test)

if __name__ == "__main__":
    main()
