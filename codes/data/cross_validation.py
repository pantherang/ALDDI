from sklearn.model_selection import KFold
import pandas as pd
kf = KFold(n_splits=5)  
fold = 0
seed = 100
all_data = pd.read_csv('./ddis_db.csv')
data_shuffled = all_data.sample(frac=1, random_state=seed)
for train_index, test_index in kf.split(data_shuffled):
    train_df = data_shuffled.iloc[train_index]
    test_df = data_shuffled.iloc[test_index]

    train_df.to_csv(f"./5-folds/train_fold_{fold}.csv", index=False)
    test_df.to_csv(f"./5-folds/test_fold_{fold}.csv", index=False)
    fold += 1
