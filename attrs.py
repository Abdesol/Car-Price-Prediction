import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def analyze_model(model, X_test, y_test):
    score = cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
    score - np.sqrt(-score)
    print(f"Score: {score}")

    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")

    model_mse = mean_squared_error(y_test, predictions)
    model_rmse = np.sqrt(model_mse)
    print(f"Root mean squared error: {model_rmse}")