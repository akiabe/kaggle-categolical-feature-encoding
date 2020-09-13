import config
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv(config.FOLD_FILE)

    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    ohe = preprocessing.OneHotEncoder()
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data)

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])
    y_train = df_train.target.values
    y_valid = df_valid.target.values

    model = linear_model.LogisticRegression()
    model.fit(x_train, y_train)
    #preds = model.predict(x_valid)
    #accuracy = metrics.accuracy_score(y_valid, preds)
    #print(f"Fold={fold}, Accuracy={accuracy}")

    valid_preds = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold={fold}, AUC={auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)