import numpy as np
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
pd.set_option('display.max_rows', 500)
from datetime import datetime


##### COPY__PASTE__LIB__BEGIN #####
basepath = os.path.abspath(os.path.dirname(os.path.abspath(sys.argv[0])) + '/..')
sys.path.append(basepath)
from edgar_playground.t6_lib import *
##### COPY__PASTE__LIB__END #####


INPUT_DIR = '../input'
# INPUT_DIR = '../work/subsample_5000'

YUKAWA_DIR = '../input/yukawa/'

# FEATURE_DIR = '.'
FEATURE_DIR = '../feature/t6'

# WORK_DIR= '.'
WORK_DIR = '../work/t6'

# OUTPUT_DIR = '.'
OUTPUT_DIR = '../work/t6'

# TYPE_WL = ['2JHH', '3JHH', '1JHC', '2JHC', '3JHC', '1JHN', '2JHN', '3JHN', ]
TYPE_WL = ['1JHN', '2JHN' ]

# TARGET_WL = ['fc', 'sd', 'pso', 'dso']
TARGET_WL = ['scalar_coupling_constant']

SEED = 55
np.random.seed(SEED)


cv_score = []
cv_score_total = 0
epoch_n = 20
verbose = 1
batch_size = 2048


# N_FOLD = {
#     '_': 5, # mint UA
# }


ATOMIC_NUMBERS = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9
}


train_dtypes = {
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}
#
# train, test, structures, contributions = t6_load_data(INPUT_DIR)
#
# train, test = t6_load_feature_criskiev(FEATURE_DIR, train, test)
#
# structures = t6_merge_yukawa(YUKAWA_DIR, structures)
#
# structures = t6_load_feature_crane(FEATURE_DIR, structures)
#
# train, test = t6_merge_structures(train, test, structures)
#
# t6_distance_feature(train, test)
#
# train, test = t6_load_feature_artgor(FEATURE_DIR, train, test)

#
# Save to and/or load from parquet
#
# t6_to_parquet(WORK_DIR, train, test, structures, contributions)

train, test, structures, contributions = t6_read_parquet(WORK_DIR)

disp_mem_usage()
print(train.shape)

#
# Edike :)
#
train, test = t6_load_feature_edgar(FEATURE_DIR, train, test)

disp_mem_usage()
print(train.shape)

#
# Load Phase 1. OOF data Mulliken charge
#
train, test = t6_load_data_mulliken_oof(WORK_DIR, train, test)

disp_mem_usage()
print(train.shape)

#
# Load Phase 2. OOF data Contributions (fc, sd, pso, dso)
#
train, test = t6_load_data_contributions_oof(WORK_DIR, train, test)

disp_mem_usage()
print(train.shape)


extra_cols = []
extra_cols += ['mulliken_charge_0', 'mulliken_charge_1']
extra_cols += ['fc', 'sd', 'pso', 'dso', 'contrib_sum']
extra_cols += ['qcut_subtype_0', 'qcut_subtype_1', 'qcut_subtype_2']
X, X_test, labels = t6_prepare_columns(train, test, good_columns_extra=extra_cols)

disp_mem_usage()
print(X.shape, X_test.shape)


def create_nn_model(input_shape):
    inp = Input(shape=(input_shape,))
    x = Dense(2048, activation="relu")(inp)
    x = BatchNormalization()(x)
    # x = Dropout(0.4)(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    out = Dense(1, activation="linear")(x)
    # out1 = Dense(2, activation="linear")(x)#mulliken charge 2
    # out2 = Dense(6, activation="linear")(x)#tensor 6(xx,yy,zz)
    # out3 = Dense(12, activation="linear")(x)#tensor 12(others) 
    # out4 = Dense(1, activation="linear")(x)#scalar_coupling_constant 
    # model = Model(inputs=inp, outputs=[out,out1,out2,out3,out4])
    model = Model(inputs=inp, outputs=[out])
    return model


def plot_history(history, label):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss for %s' % label)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    _ = plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# Set up GPU preferences
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 2})
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config=config)
K.set_session(sess)


# Set to True if we want to train from scratch.  False will reuse saved models as a starting point.
retrain = True

start_time = datetime.now()
test_prediction = np.zeros(len(X_test))
# input_features = ['atom_2', 'atom_3', 'atom_4', 'atom_5', 'atom_6', 'atom_7',
#                   'atom_8', 'atom_9', 'atom_10', 'd_1_0', 'd_2_0', 'd_2_1', 'd_3_0',
#                   'd_3_1', 'd_3_2', 'd_4_0', 'd_4_1', 'd_4_2', 'd_4_3', 'd_5_0',
#                   'd_5_1', 'd_5_2', 'd_5_3', 'd_6_0', 'd_6_1', 'd_6_2', 'd_6_3',
#                   'd_7_0', 'd_7_1', 'd_7_2', 'd_7_3', 'd_8_0', 'd_8_1', 'd_8_2',
#                   'd_8_3', 'd_9_0', 'd_9_1', 'd_9_2', 'd_9_3', 'd_10_0', 'd_10_1', 'd_10_2',
#                   'd_10_3']

# Loop through each molecule type
for type_name in TYPE_WL:
    t = labels['type'].transform([type_name])[0]

    model_name_wrt = ('../work/keras-neural-net-and-distance-features-molecule_model_%s.hdf5' % type_name)
    print('Training %s' % type_name, 'out of', TYPE_WL, '\n')

    df_train_ = X[train['type'] == t]
    df_test_ = X_test[test['type'] == t]
    df_train_  = df_train_.fillna(0)
    df_test_  = df_test_.fillna(0)
    input_features = list(X.columns)

    print(df_train_.dtypes.T)
    print(df_train_.describe().T)

    disp_mem_usage()
    print(df_train_.shape)
    print(df_test_.shape)

    # Standard Scaler from sklearn does seem to work better here than other Scalers
    input_data = StandardScaler().fit_transform(
        pd.concat([df_train_.loc[:, input_features], df_test_.loc[:, input_features]]))
    # input_data=StandardScaler().fit_transform(df_train_.loc[:,input_features])
    target_data = train.loc[train['type'] == t, "scalar_coupling_constant"].values

    disp_mem_usage()
    print(input_data.shape)
    print(target_data.shape)

    # Simple split to provide us a validation set to do our CV checks with
    train_index, cv_index = train_test_split(np.arange(len(df_train_)), random_state=SEED, test_size=0.1)
    # Split all our input and targets by train and cv indexes
    train_target = target_data[train_index]
    cv_target = target_data[cv_index]
    train_input = input_data[train_index]
    cv_input = input_data[cv_index]
    test_input = input_data[len(df_train_):, :]

    disp_mem_usage()
    print(input_data.shape)
    print(train_input.shape)
    print(train_target.shape)
    print(cv_input.shape)
    print(cv_target.shape)

    # Build the Neural Net
    nn_model = create_nn_model(train_input.shape[1])

    # If retrain==False, then we load a previous saved model as a starting point.
    if not retrain:
        # nn_model = load_model(model_name_rd)
        pass

    nn_model.compile(loss='mae', optimizer=Adam())  # , metrics=[auc])

    # Callback for Early Stopping... May want to raise the min_delta for small numbers of epochs
    es = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=40, verbose=1, mode='auto',
                                 restore_best_weights=True)
    # Callback for Reducing the Learning Rate... when the monitor levels out for 'patience' epochs, then the LR is reduced
    rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, min_lr=1e-6, mode='auto', verbose=1)
    # Save the best value of the model for future use
    sv_mod = callbacks.ModelCheckpoint(model_name_wrt, monitor='val_loss', save_best_only=True, period=1)
    history = nn_model.fit(train_input, [train_target],
                           validation_data=(cv_input, [cv_target]),
                           callbacks=[es, rlr, sv_mod], epochs=epoch_n, batch_size=batch_size, verbose=verbose)

    cv_predict = nn_model.predict(cv_input)

    # plot_history(history, mol_type)
    accuracy = np.mean(np.abs(cv_target - cv_predict[:, 0]))
    print(np.log(accuracy))
    cv_score.append(np.log(accuracy))
    cv_score_total += np.log(accuracy)

    # Predict on the test data set using our trained model
    test_predict = nn_model.predict(test_input)

    # for each molecule type we'll grab the predicted values
    test_prediction[test["type"] == t] = test_predict[:, 0]
    K.clear_session()

cv_score_total /= len(TYPE_WL)

print('Total training time: ', datetime.now() - start_time)

i = 0
for type_name in TYPE_WL:
    print(type_name, ": cv score is ", cv_score[i])
    i += 1
print("total cv score is", cv_score_total)

print(type(test_prediction))
print(test_prediction.shape)
print(test_prediction)


def submits(predictions):
    submit = t6_load_submissions(INPUT_DIR)
    submit["scalar_coupling_constant"] = predictions
    submit.to_csv(f'{OUTPUT_DIR}/t6c_scc.csv', index=False)


submits(test_prediction)

# Add more layers to get a better score! However,maybe,features are really more important than algorithms...
