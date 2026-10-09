# Student Score Predictor

A linear regression model built from scratch in pure Python (no libraries) that predicts exam score from daily study hours.

## What it does
- Loads hours/score data from `data.csv`
- Trains `score = m * hours + b` using gradient descent on mean squared error
- Verifies the result against the exact least-squares formula
- Reports MSE and R^2, and predicts scores for new inputs

## Run
    python predictor.py
    python predictor.py 6.5

## Concepts used
Linear regression, gradient descent, mean squared error, R^2, partial derivatives.

## Note
`data.csv` is small sample data for learning. Replace it with real data to use it seriously.

## Ideas to extend
- Add a plot with matplotlib
- Add a second feature (e.g. sleep hours) and use NumPy
- Compare against scikit-learn's LinearRegression
