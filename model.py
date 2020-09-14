import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

A = np.load('penalty_matrix.npy')
def score(y_true, y_pred):
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]

df = pd.read_pickle('initial_training_data')
print(df.describe())

print(len(df))
df.dropna(inplace=True)
print(len(df))
df_clean = df.loc[df['BADHOLE_CAL']== 0]
print(len(df_clean))

X = df_clean[['VSHALE', 'RHOB_FIX', 'DTC', 'TVD']]
y = df_clean['FORCE_2020_LITHOFACIES_LITHOLOGY']


X_train, X_test, y_train, y_test = train_test_split(X, y)

print(X_train)
print(y_train)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)

score = score(y_test.values, y_pred_test)
print(score)

pickle.dump(model, open('model_excl_badhole.pkl', 'wb'))