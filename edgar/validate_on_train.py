import pandas as pd
import sklearn.metrics
import math

valid = pd.read_csv('../input/train.csv', index_col='id')[['type', 'scalar_coupling_constant']]
pred = pd.read_csv('../work/median_by_type_on_train.csv', index_col='id')[['scalar_coupling_constant']]

error = 0
for t in valid['type'].unique():
    filter = valid['type'] == t
    log_error = math.log(sklearn.metrics.mean_absolute_error(valid[filter]['scalar_coupling_constant'], pred[filter]['scalar_coupling_constant']))
    print(t, '%.4f' % log_error)
    error += log_error

error /= len(valid['type'].unique())
print( '%.4f' % error)