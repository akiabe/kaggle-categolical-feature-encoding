import config
import warnings
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

    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train[features].values
    x_valid = df_valid[features].values
    y_train = df_train.target.values
    y_valid = df_valid.target.values

    model = linear_model.LogisticRegression()
    model.fit(x_train, y_train)
    preds = model.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    #valid_preds = model.predict_proba(x_valid)[:, 1]
    #auc = metrics.roc_auc_score(y_valid, valid_preds)
    #print(f"Fold={fold}, AUC={auc}")

    warnings.filterwarnings('ignore')

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)