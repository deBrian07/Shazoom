import json
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 1) Load feature CSV
df = pd.read_csv("hash_features.csv")

# 2) One-hot encode the family feature
ohe = OneHotEncoder(sparse=False)
fam_ohe = ohe.fit_transform(df[["family"]])
fam_cols = ohe.get_feature_names_out(["family"])
X_num = df[["df", "delta_f"]].copy()
X = pd.concat([
    X_num.reset_index(drop=True),
    pd.DataFrame(fam_ohe, columns=fam_cols)
], axis=1)

y = df["label"]

# 3) Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4) Build LightGBM dataset
dtrain = lgb.Dataset(X_train, label=y_train)
dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

# 5) Define hyperparameters (tune via grid or Bayesian)
params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": 6,
    "verbose": -1
}

# 6) Train with early stopping
bst = lgb.train(
    params,
    dtrain,
    num_boost_round=500,
    valid_sets=[dval],
    early_stopping_rounds=50
)

# 7) Predict a weight for *every* unique hash in your catalog
#    (Assumes your CSV contained one row per hash–feature; if you had duplicates,
#     drop duplicates first.)
weights = {}
for idx, row in df.drop_duplicates(subset=["hash"]).iterrows():
    feats = []
    # numeric
    feats.append(row["df"])
    feats.append(row["delta_f"])
    # one-hot family
    for fam in fam_cols:
        feats.append(1.0 if row["family"] == fam.split("_",1)[1] else 0.0)
    # predict probability (or raw score)
    w = bst.predict([feats])[0]
    weights[row["hash"]] = float(w)

# 8) Dump to JSON
with open("hash_weights.json", "w") as f:
    json.dump(weights, f, indent=2)

print(f"Exported weights for {len(weights)} hashes.")
