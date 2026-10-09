"""Student Score Predictor: linear regression built from scratch in pure Python.

Usage:
    python predictor.py          # train, show metrics, predict a few examples
    python predictor.py 6.5      # predict the score for 6.5 study hours
"""
import csv
import sys


def load_data(path="data.csv"):
    hours, scores = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            hours.append(float(row["hours"]))
            scores.append(float(row["score"]))
    return hours, scores


def gradient_descent(x, y, lr=0.01, epochs=20000):
    """Fit y = m*x + b by minimising mean squared error."""
    m, b, n = 0.0, 0.0, len(x)
    for _ in range(epochs):
        dm = (-2 / n) * sum(xi * (yi - (m * xi + b)) for xi, yi in zip(x, y))
        db = (-2 / n) * sum(yi - (m * xi + b) for xi, yi in zip(x, y))
        m -= lr * dm
        b -= lr * db
    return m, b


def closed_form(x, y):
    """Exact least-squares solution, used to check gradient descent."""
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    m = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / sum(
        (xi - mean_x) ** 2 for xi in x
    )
    return m, mean_y - m * mean_x


def evaluate(x, y, m, b):
    preds = [m * xi + b for xi in x]
    mse = sum((yi - p) ** 2 for yi, p in zip(y, preds)) / len(y)
    mean_y = sum(y) / len(y)
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - p) ** 2 for yi, p in zip(y, preds))
    return mse, 1 - ss_res / ss_tot  # (MSE, R^2)


def main():
    x, y = load_data()
    m, b = gradient_descent(x, y)
    m_exact, b_exact = closed_form(x, y)
    mse, r2 = evaluate(x, y, m, b)

    print(f"Gradient descent : score = {m:.3f} * hours + {b:.3f}")
    print(f"Closed form      : score = {m_exact:.3f} * hours + {b_exact:.3f}")
    print(f"MSE = {mse:.2f}, R^2 = {r2:.4f}")

    hours_list = [float(a) for a in sys.argv[1:]] or [2, 5, 8]
    for h in hours_list:
        print(f"Predicted score for {h} hours: {min(100, m * h + b):.1f}")


if __name__ == "__main__":
    main()
