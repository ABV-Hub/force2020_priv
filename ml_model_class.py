import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import  RandomForestRegressor
import matplotlib.pyplot as plt

import missingno

class Model(object):
    def __init__(self, dataframe):
        self.df = dataframe
        self.A = np.load('penalty_matrix.npy')

        self.features =  ['VSHALE', 'RHOB_COMBINED', 'DTC_FG', 'TVD', 'NPHI_COMBINED', 'PEF_COMBINED', 'LITH_M', 'LITH_N', 'RDEP', 'RMED', 'BAAT GP.', 'BOKNFJORD GP.', 'CROMER KNOLL GP.', 'DUNLIN GP.', 'HEGRE GP.', 'NORDLAND GP.', 'ROGALAND GP.', 'ROTLIEGENDES GP.', 'SHETLAND GP.', 'TYNE GP.', 'VESTLAND GP.', 'VIKING GP.', 'ZECHSTEIN GP.']
    
    def test(self):
        print(self.df.head())
    
    def train_init(self):
        self.workingdf = self.df.loc[:,['WELL', 'GROUP', 'DEPTH_MD', 'CALI', 'RDEP', 'RMED', 'DRHO', 'GR', 'RHOB', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'Z_LOC', 'X_LOC', 'Y_LOC', 'FORCE_2020_LITHOFACIES_CONFIDENCE', 'FORCE_2020_LITHOFACIES_LITHOLOGY']]
        self.encoder()
        print(self.workingdf.columns)
        self.lithology_conversion(0)
        print('Dropping lithofacies with confidence value of 3...')
        self.workingdf.drop(self.workingdf[self.workingdf.FORCE_2020_LITHOFACIES_CONFIDENCE == 3].index, inplace=True)
        self.normalise_gr()
        self.normalise_pef()
        self.calculate_vol_shale()
        self.calculate_synth_bitsize()
        self.create_tvd()
        self.workingdf.loc[(self.workingdf.DTC < 40), 'DTC']= np.nan
        self.workingdf.loc[(self.workingdf.DTC >= 190), 'DTC']= np.nan
        self.workingdf.loc[(self.workingdf.RDEP < 0), 'RDEP']= np.nan
        self.workingdf.loc[(self.workingdf.RMED < 0), 'RMED']= np.nan
        self.workingdf['DTC_FG'] = self.workingdf['DTC'].fillna(method='ffill')
        self.rhob_fix()
        self.nphi_fix()
        self.pef_fix()
        self.m_and_n()
        print(self.workingdf.info())
        print(self.workingdf.describe())
        self.workingdf.to_pickle('clean_training_data.pkl')
        self.apply_scaler()
        self.workingdf.to_pickle('clean_training_data_scaled.pkl')


    def train_init_from_file(self):
        # TODO: If training data has already been created we can load the pickle file
        pass
    
    def m_and_n(self):
        self.workingdf['LITH_M'] = ((189 - self.workingdf['DTC_FG'])/ (self.workingdf['RHOB_COMBINED'] - 1)) * 0.01
        self.workingdf['LITH_N'] = (1 - self.workingdf['NPHI_COMBINED']) / (self.workingdf['RHOB_COMBINED'] - 1)

    def test_init(self):
        self.workingdf = self.df.loc[:,['WELL', 'GROUP', 'DEPTH_MD', 'CALI', 'RDEP', 'RMED', 'DRHO', 'GR', 'RHOB', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'Z_LOC', 'X_LOC', 'Y_LOC']]
        # self.lithology_conversion(0)
        self.encoder()
        self.normalise_gr()
        self.normalise_pef()
        self.calculate_vol_shale()
        self.calculate_synth_bitsize()
        self.create_tvd()
        self.workingdf.loc[(self.workingdf.DTC < 40), 'DTC']= np.nan
        self.workingdf.loc[(self.workingdf.DTC >= 190), 'DTC']= np.nan
        self.workingdf.loc[(self.workingdf.RDEP < 0), 'RDEP']= np.nan
        self.workingdf.loc[(self.workingdf.RMED < 0), 'RMED']= np.nan
        self.workingdf['DTC_FG'] = self.workingdf['DTC'].fillna(method='ffill')
        self.rhob_fix()
        self.nphi_fix()
        self.pef_fix()
        self.m_and_n()
        self.apply_scaler()
        print(self.workingdf.info())
        print(self.workingdf.describe())
    
    def build_model(self):
        x_features = self.features
        print('Creating training set.....')
        training_data = self.workingdf.loc[:,['VSHALE', 'RHOB_COMBINED', 'DTC_FG', 'TVD', 'NPHI_COMBINED', 'PEF_COMBINED', 'LITH_M', 'LITH_N', 'RDEP', 'RMED', 'BAAT GP.', 'BOKNFJORD GP.', 'CROMER KNOLL GP.', 'DUNLIN GP.', 'HEGRE GP.', 'NORDLAND GP.', 'ROGALAND GP.', 'ROTLIEGENDES GP.', 'SHETLAND GP.', 'TYNE GP.', 'VESTLAND GP.', 'VIKING GP.', 'ZECHSTEIN GP.', 'FORCE_2020_LITHOFACIES_LITHOLOGY']]
        training_data.to_pickle('model_training_data')
        
        X = training_data[x_features]
        y = training_data['FORCE_2020_LITHOFACIES_LITHOLOGY']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print('Fitting the classifier.....')
        clf = XGBClassifier()
        clf.fit(X_train, y_train)

        print('KFold Validation')
        
        # Cross fold validation:
        kfold = StratifiedKFold(n_splits=3, random_state=42)
        results = cross_val_score(clf, X, y, cv=kfold)
        print(f'Accuracy from KFOLD: {results.mean()*100}, {results.std()*100}')


        print('Predicting on X_test.....')
        y_pred_test = clf.predict(X_test)

        print('Scoring the classifier.....')
        accuracy = accuracy_score(y_test, y_pred_test)
        xeek_score = self.score(y_test.values, y_pred_test)
        print(f'Accuracy: {accuracy * 100}%')
        print(f'XEEK Score: {xeek_score}')

        
        pickle.dump(clf, open('model.pkl', 'wb'))

    def create_tvd(self):
        """
        Fill in gaps in the TVD curve. First fill in values using First One Carried Backward,
        then fill in any remaining gaps with Last One Carried Forward.
        """
        self.workingdf['TVD'] = self.workingdf['Z_LOC'] * -1
        self.workingdf['TVD'].fillna(method='bfill', inplace=True)
        self.workingdf['TVD'].fillna(method='ffill', inplace=True)

    def train_predict(self):
        pass

    def test_predict(self):
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

        #df = pd.read_csv()
        model = pickle.load(open('model.pkl', 'rb'))

        # print(df.head())
        # print(model)


        open_test_features = self.workingdf.loc[:,['VSHALE', 'RHOB_COMBINED', 'DTC_FG',  'TVD', 'NPHI_COMBINED', 'PEF_COMBINED', 'LITH_M', 'LITH_N', 'RDEP', 'RMED','BAAT GP.', 'BOKNFJORD GP.', 'CROMER KNOLL GP.', 'DUNLIN GP.', 'HEGRE GP.', 'NORDLAND GP.', 'ROGALAND GP.', 'ROTLIEGENDES GP.', 'SHETLAND GP.', 'TYNE GP.', 'VESTLAND GP.', 'VIKING GP.', 'ZECHSTEIN GP.']]
        print(open_test_features.head())

        test_prediction = model.predict(open_test_features)
        print(open_test_features)

        category_to_lithology = {y:x for x,y in lithology_numbers.items()}
        test_prediction_for_submission = np.vectorize(category_to_lithology.get)(test_prediction)

        np.savetxt('test_predictions.csv', test_prediction_for_submission, header='lithology', comments='', fmt='%i')

    def apply_scaler(self):
        print('Applying Scaler.....')
        col_names = ['VSHALE', 'TVD', 'NPHI_COMBINED', 'PEF_COMBINED', 'RHOB_COMBINED', 'DTC_FG', 'LITH_M', 'LITH_N', 'RDEP', 'RMED']
        features = self.workingdf[col_names]
        scaler=StandardScaler().fit(features.values)
        features = scaler.transform(features.values)
        self.workingdf[col_names] = features
        print(self.workingdf)
        print('Applying Scaler Complete')
        pickle.dump(scaler, open('scaler.pkl', 'wb'))

    def encoder(self):
        # encoding_df = self.workingdf.loc[:,['WELL','GROUP']]
        # dummies = pd.get_dummies(encoding_df.GROUP)
        # self.workingdf = pd.concat([self.workingdf, dummies.reindex(self.workingdf.index)], axis=1)
        self.workingdf = pd.concat([self.workingdf, pd.get_dummies(self.workingdf.GROUP)], axis=1)

    def rhob_fix(self):
        print('Fixing RHOB.....')
        model_RFR = pickle.load(open('RHOB_RFR_model.pkl', 'rb'))
        X_features_rhob = self.workingdf.loc[:,['TVD', 'VSHALE', 'DTC_FG']].copy()
        
        col_names = ['DTC_FG', 'TVD']
        features_rhob = X_features_rhob[col_names]
        scaler=StandardScaler().fit(features_rhob.values)
        features_rhob = scaler.transform(features_rhob.values)

        full_rhob_pred = model_RFR.predict(X_features_rhob)
        self.workingdf['RHOB_SYNTH'] = full_rhob_pred
        self.workingdf['RHOB_COMBINED'] = self.workingdf['RHOB']
        self.workingdf.loc[self.workingdf.DIFF_CAL > 3, "RHOB_COMBINED"]=self.workingdf['RHOB_SYNTH']
        self.workingdf['RHOB_COMBINED'].fillna(self.workingdf['RHOB_SYNTH'], inplace=True)

    def pef_fix(self):
        print('Fixing PEF.....')
        PEF_model_RFR = pickle.load(open('model_PEF_RFR.pkl', 'rb'))
        X_features_pef = self.workingdf.loc[:,['DTC_FG', 'TVD', 'VSHALE', 'RHOB_COMBINED', 'NPHI_COMBINED']].copy()
        
        col_names = ['DTC_FG', 'TVD', 'VSHALE', 'RHOB_COMBINED', 'NPHI_COMBINED']
        features_pef = X_features_pef[col_names]
        scaler=StandardScaler().fit(features_pef.values)
        features_pef = scaler.transform(features_pef.values)

        full_pef_pred = PEF_model_RFR.predict(X_features_pef)
        self.workingdf['PEF_SYNTH'] = full_pef_pred
        self.workingdf['PEF_COMBINED'] = self.workingdf['PEF_NORM']
        self.workingdf.loc[self.workingdf.PEF_NORM < 1, "PEF_COMBINED"]=self.workingdf['PEF_SYNTH']
        self.workingdf.loc[self.workingdf.PEF_NORM > 15, "PEF_COMBINED"]=self.workingdf['PEF_SYNTH']

        self.workingdf['PEF_COMBINED'].fillna(self.workingdf['PEF_SYNTH'], inplace=True)

    def nphi_fix(self):
        print('Fixing NPHI.....')
        NPHI_model_RFR = pickle.load(open('NPHI_RFR_model.pkl', 'rb'))
        X_features_nphi = self.workingdf.loc[:,['DTC_FG', 'TVD', 'VSHALE', 'RHOB_COMBINED']].copy()
        
        col_names = ['DTC_FG', 'TVD', 'VSHALE', 'RHOB_COMBINED']
        features_nphi = X_features_nphi[col_names]
        scaler=StandardScaler().fit(features_nphi.values)
        features_nphi = scaler.transform(features_nphi.values)

        full_nphi_pred = NPHI_model_RFR.predict(X_features_nphi)
        self.workingdf['NPHI_SYNTH'] = full_nphi_pred
        self.workingdf['NPHI_COMBINED'] = self.workingdf['NPHI']
        self.workingdf.loc[self.workingdf.DIFF_CAL > 3, "NPHI_COMBINED"]=self.workingdf['NPHI_SYNTH']
        self.workingdf['NPHI_COMBINED'].fillna(self.workingdf['NPHI_SYNTH'], inplace=True)

    def normalise_gr(self):
        percentile_95 = self.workingdf.groupby('WELL')['GR'].quantile(0.95)
        self.workingdf ['95_PERC'] = self.workingdf['WELL'].map(percentile_95)
        percentile_05 = self.workingdf.groupby('WELL')['GR'].quantile(0.05)
        self.workingdf ['05_PERC'] = self.workingdf['WELL'].map(percentile_05)

        # Key Well High and Low
        # 35/9-5 shows a nice distribution for shale and sand
        print('Normalising Gamma Ray.....')
        key_well_low = 50.822391
        key_well_high = 131.688494
        self.workingdf['GR_NORM'] = self.workingdf.apply(lambda x: self.normalise(x['GR'], key_well_low, key_well_high, x['05_PERC'], x['95_PERC']), axis=1)
        print('Normalising Gamma Ray Complete!')

    def normalise_pef(self):
        print('Calculatig PEF Percentiles.....')
        pef_percentile_95 = self.workingdf.groupby('WELL')['PEF'].quantile(0.95)
        self.workingdf ['95_PERC_PEF'] = self.workingdf['WELL'].map(pef_percentile_95)
        pef_percentile_05 = self.workingdf.groupby('WELL')['PEF'].quantile(0.05)
        self.workingdf ['05_PERC_PEF'] = self.workingdf['WELL'].map(pef_percentile_05)

        # Key Well High and Low
        # Taking 25/2-7 as a key well
        print('Normalising PEF.....')
        key_well_low = 1.603071
        key_well_high = 6.22167
        self.workingdf['PEF_NORM'] = self.workingdf.apply(lambda x: self.normalise(x['PEF'], key_well_low, key_well_high, x['05_PERC_PEF'], x['95_PERC_PEF']), axis=1)
        print('Normalising PEF Complete!')

    def normalise(self, curve, ref_low, ref_high, well_low, well_high):
        norm = ref_low + ((ref_high - ref_low) * ((curve - well_low) / (well_high - well_low)))
        return norm

    def calculate_vol_shale(self):
        print(f'Calculating VShale......')
        self.workingdf['VSHALE'] = self.workingdf.apply(lambda x: self.vol_shale(50.822391, 131.688494, x['GR_NORM']), axis=1)
        print(f'Calculating VShale Complete!')

    def calculate_synth_bitsize(self):
        # Fixing bitsize due to "odd" bitsize values, some look interpolated others don't make sense
        print('Fixing Existing Bitsize (BS) Curve.....')
        self.workingdf['BS_FIX'] = self.workingdf.apply(lambda x: self.bs_fix(x.loc['BS']), axis=1)

        # This is an attempt to create a synthetic bitsize curve based on a rolling average of the CALI curve
        print('Creating Synthetic Bitsize From CALI.....')
        self.workingdf['CALI_R_MEAN'] = self.workingdf['CALI'].rolling(10).mean()
        self.workingdf['BS_FROM_CALI'] = self.workingdf.apply(lambda x: self.bs_fix(x.loc['CALI']), axis=1)

        # Combine the Bitsize curves
        print('Combining Existing Bitsize and Fixed Bitsize Curves....')
        self.workingdf['BS_COMB'] = self.workingdf['BS_FIX']
        self.workingdf['BS_COMB'].fillna(self.workingdf['BS_FROM_CALI'], inplace=True)

        # Create a new differential caliper. Existing one had poor coverage
        print('Calculating Diff CAL....')
        self.workingdf['DIFF_CAL'] = self.workingdf['CALI'] - self.workingdf['BS_COMB']
        print('Syntehtic BS and Differential Caliper Calculations Complete!')
    
    def bad_hole_flag(self):
        print('Calculating bad hole flag')

    def bs_fix(self, bitsize):
        #standard_bs_vals = [26, 17.5, 17, 14.75, 12.5, 12.25, 9.875, 8.5, 8.375, 6.5, 6]
        if bitsize >= 25 and bitsize <= 27:
            return 26
        elif bitsize >= 17.3 and bitsize <= 24:
            return 17.5
        elif bitsize >= 16.0 and bitsize < 17.3:
            return 17
        elif bitsize >= 14 and bitsize <= 15:
            return 14.75
        elif bitsize >= 12.3 and bitsize <= 13:
            return 12.5
        elif bitsize >= 11.8 and bitsize < 12.3:
            return 12.25
        elif bitsize >= 9.4 and bitsize <= 11.5:
            return 9.875
        elif bitsize >= 8.3 and bitsize < 9.4:
            return 8.5
        elif bitsize >= 8 and bitsize < 8.3:
            return 8.25
        elif bitsize >= 6.3 and bitsize <=7:
            return 6.75
        elif bitsize >= 5.8 and bitsize < 6.3:
            return 6
        else:
            return np.nan

    def vol_shale(self, clean, shale, curve):
        vshale = (curve - clean) / (shale - clean)
        if vshale > 1:
            vshale = 1

        if vshale < 0:
            vshale = 0
        
        return vshale

    def lithology_conversion(self, direction='0'):
        # Lithology numbers could be extracted out to a class level dictionary
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
        if direction == 0: # NPD code to 0-12
            self.workingdf['FORCE_2020_LITHOFACIES_LITHOLOGY'] = self.workingdf['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(lithology_numbers)
        elif direction == 1: # 0-12 to NPD code NOTE: NOT TESTED
            category_to_lithology = {y:x for x,y in lithology_numbers.items()}

    def score(self, y_true, y_pred):
        S = 0.0
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        for i in range(0, y_true.shape[0]):
            S -= self.A[y_true[i], y_pred[i]]
        return S/y_true.shape[0]

import pandas as pd

train_data = pd.read_csv('data/train.csv', sep=';')
test_data = pd.read_csv('data/test.csv', sep=';')


train = False

if train:
    print('Training Mode Selected')
    train_model = Model(train_data)
    train_model.train_init()
    train_model.build_model()
else:
    print('Testing Mode Selected')
    test_model = Model(test_data)
    test_model.test_init()
    test_model.test_predict()
