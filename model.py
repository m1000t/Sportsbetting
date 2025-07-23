import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def train_improved_model(path="cleaned_data.csv"):
    df = pd.read_csv(path)

    # Create new features
    df["implied_prob_diff"] = df["home_implied_prob"] - df["away_implied_prob"]

    # Use these features
    feature_cols = ["home_implied_prob", "away_implied_prob", "spread", "total", "implied_prob_diff"]

    # Drop rows with missing data in these features
    df = df.dropna(subset=feature_cols + ["home_win"])

    X = df[feature_cols]
    y = df["home_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression with GridSearchCV for hyperparameter tuning
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    lr = LogisticRegression(max_iter=1000, solver="liblinear")
    grid = GridSearchCV(lr, param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"Best C: {grid.best_params_['C']}")

    # Predictions and accuracy
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    return best_model, X_test, y_test, df.loc[X_test.index]

if __name__ == "__main__":
    train_improved_model()
