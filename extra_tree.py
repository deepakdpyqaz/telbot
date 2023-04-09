import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import multiprocessing
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from telegram import send_to_telegram, send_to_telegram_document
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    filename="extra_tree.log",
    filemode="w",
)
DATA_DIR = "./data"
parameters_file = "./info/extraTreeRandomized.csv"


def create_log(*args):
    log = f"Extra tree: {' '.join(map(str,args))}"
    logging.info(log)
    print(log)
    try:
        send_to_telegram(log)
    except Exception as e:
        logging.error(str(e))
        print(e)


# Read data
csvs = Path(DATA_DIR).glob("*.csv")
tamilNaduData = pd.concat([pd.read_csv(str(csv)) for csv in csvs])

y = tamilNaduData["Wind Speed"]
x = tamilNaduData.drop(["Wind Speed"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=123
)

create_log("Training dataset shape:", X_train.shape, y_train.shape)
create_log("Testing dataset shape:", X_test.shape, y_test.shape)


parameters_csv = pd.read_csv(parameters_file)
parameters_csv.replace(np.nan, None, regex=True, inplace=True)
parameters = []
for idx in parameters_csv.index:
    d = parameters_csv.iloc[idx, :].to_dict()
    new_d = {}
    for idx in d:
        try:
            if type(d[idx])==str and d[idx].replace(".","",1).isnumeric():
                d[idx] = eval(d[idx])
            new_d[idx] = [(d[idx])]
        except Exception as e:
            traceback.print_exc()
            exit(1)
    parameters.append(new_d)

def evaluate(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae, "mape": mape}


def make_predictions(params):
    try:
        rf = ExtraTreesRegressor()

        rf_random = GridSearchCV(rf, params, cv=3, error_score=np.nan)

        start = time.time()
        rf_random.fit(X_train, y_train)
        stop = time.time()
        y_pred = rf_random.predict(X_test)
        evaluation = evaluate(y_test, y_pred)
        evaluation["time"] = stop - start
        np.save(f"extra_tree_tn/{rf_random.best_score_}.npy", y_pred)
        create_log(f"Done for {len(os.listdir('extra_tree_tn'))}")
        return evaluation
    except Exception as e:
        create_log(str(e))
        return {"r2": 0, "mse": 0, "rmse": 0, "mae": 0, "mape": 0, "time": 0}


if __name__ == "__main__":
    os.mkdir("extra_tree_tn")
    op = []
    create_log(f"Total {len(parameters)} models")

    with multiprocessing.Pool(4) as pool:
        op = pool.map(make_predictions, parameters)
    df = pd.DataFrame(op)
    df.to_csv("extra_tree_tn/evaluation_extra_tree.csv", index=False)
    send_to_telegram_document("extra_tree_tn/evaluation_extra_tree.csv")
    create_log("Done")
