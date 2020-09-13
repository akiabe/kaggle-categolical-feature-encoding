import pandas as pd

df = pd.read_csv("../input/train.csv")

features = [f for f in df.columns if f not in ("id", "target", "kfold")]
print(features)

print(df.ord_2.value_counts())

print("labeld encoding...")
print("mapping...")
mapping = {
    "Freezing": 0,
    "Warm": 1,
    "Cold": 2,
    "Boiling Hot": 3,
    "Hot": 4,
    "Lava Hot": 5
}
df.loc[:, "ord_2"] = df.ord_2.map(mapping)
print(df.ord_2.value_counts())






