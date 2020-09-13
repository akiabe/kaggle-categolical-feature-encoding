import config
import pandas as pd
from sklearn import model_selection

df = pd.read_csv(config.TRAIN_FILE)

df["kfold"] = -1

df = df.sample(frac=1).reset_index(drop=True)
y = df.target.values

kf = model_selection.StratifiedKFold(n_splits=5)
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    print("train_idx and kfold")
    print()
    print(df.loc[t_, 'kfold'])
    print()
    print("val_idx and kfold")
    print()
    print(df.loc[v_, 'kfold'])
    print()
    df.loc[v_, 'kfold'] = f
    print("val_idx and replace kfold number")
    print()
    print(df.loc[v_, 'kfold'])
    print()

df.to_csv(
    "../input/train_folds.csv",
    index=False
)