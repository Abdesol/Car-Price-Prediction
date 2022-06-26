import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def analyze_model(model, X_test, y_test):
    print(f"Score: {model.score(X_test, y_test)}")

    predictions = model.predict(X_test)
    print(f"Predictions: {predictions}")

    model_mse = mean_squared_error(y_test, predictions)
    model_rmse = np.sqrt(model_mse)
    print(f"Root mean squared error: {model_rmse}")