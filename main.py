import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os
from pathlib import Path
import multiprocessing
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


DATA_DIR = "/content/drive/MyDrive/Wind Speed/Madurai, Tamil Nadu Data"
parameters_file = "/content/tamil_nadu/randomForestRandomized"

# Read data
csvs = Path(DATA_DIR).glob("*.csv")
tamilNaduData = pd.concat([pd.read_csv(str(csv)) for csv in csvs])

y = tamilNaduData["Wind Speed"]
x = tamilNaduData.drop(['Wind Speed'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=123)

print('Training dataset shape:', X_train.shape, y_train.shape)
print('Testing dataset shape:', X_test.shape, y_test.shape)


parameters = pd.read_csv(parameters_file,index_col=0)
parameters = parameters["params"]
def make_param_list(item):
  d = eval(item)
  new_d = {}
  for idx in d:
    new_d[idx] = [d[idx]]
  return new_d
parameters = parameters.apply(make_param_list)


def evaluate(y_test,y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return {
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }

def make_predictions(params):
    rf = RandomForestRegressor()

    rf_random = GridSearchCV(rf, params,  cv = 3, error_score=np.nan)

    start = time.time()
    rf_random.fit(X_train, y_train)
    stop = time.time()
    y_pred = rf_random.predict(X_test)
    evaluation = evaluate(y_test, y_pred)
    evaluation["time"] = stop - start
    np.save(f"random_forest_tn/{rf_random.best_score_}.npy",y_pred)
    return evaluation

if __name__ == '__main__':
    os.mkdir("random_forest_tn")
    with Pool(4) as p:
        op = p.map(make_predictions, parameters)
    df = pd.DataFrame(op)
    df.to_csv("random_forest_tn/evaluation.csv")
