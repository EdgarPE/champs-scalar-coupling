import pandas as pd

train = pd.read_csv('../input/train.csv', index_col='id')
# train = pd.read_csv('../work/train_sample.csv', index_col='id')
train = train[['type', 'scalar_coupling_constant']]

# test = pd.read_csv('../input/test.csv', index_col='id')
test = train.copy()
test = test[['type']]
test['scalar_coupling_constant'] = None

for t in train['type'].unique():
    median = train.scalar_coupling_constant[train.type == t].median()
    print(t, median)
    test['scalar_coupling_constant'][test.type == t] = median

submission = test[['scalar_coupling_constant']]
submission.to_csv('../work/median_by_type_on_train.csv')
