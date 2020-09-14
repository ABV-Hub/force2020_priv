import pickle
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

lithology_numbers = {30000: 0,
                 65030: 1,
                 65000: 2,
                 80000: 3,
                 74000: 4,
                 70000: 5,
                 70032: 6,
                 88000: 7,
                 86000: 8,
                 99000: 9,
                 90000: 10,
                 93000: 11}

df = pd.read_pickle('initial_test_data')
model = pickle.load(open('model_excl_badhole.pkl', 'rb'))

print(df.head())
print(model)


open_test_features = df[['VSHALE', 'RHOB_FIX', 'DTC', 'TVD']]

test_prediction = model.predict(open_test_features)
print(open_test_features)

category_to_lithology = {y:x for x,y in lithology_numbers.items()}
test_prediction_for_submission = np.vectorize(category_to_lithology.get)(test_prediction)

np.savetxt('test_predictions.csv', test_prediction_for_submission, header='lithology', comments='', fmt='%i')