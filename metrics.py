import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def regression_metrics(y, prediction):
    return {
        "R2": r2_score(y, prediction),
        "RMSE": np.sqrt(mean_squared_error(y, prediction)),
        "MAE": mean_absolute_error(y, prediction),
    }
