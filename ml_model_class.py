import pandas as pd
import numpy as np

class Model(object):
    def __init__(self, dataframe):
        self.df = dataframe
        self.A = np.load('penalty_matrix.npy')
    
    def test(self):
        print(self.df.head())
    
    def train_init(self):
        self.workingdf = self.df.loc[:,['WELL', 'DEPTH_MD', 'CALI', 'RDEP', 'RMED', 'DRHO', 'GR', 'RHOB', 'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'Z_LOC', 'X_LOC', 'Y_LOC', 'FORCE_2020_LITHOFACIES_CONFIDENCE', 'FORCE_2020_LITHOFACIES_LITHOLOGY']]
        self.lithology_conversion(0)
        self.normalise_gr()
        self.calculate_vol_shale()
        self.calculate_synth_bitsize()
        print(self.workingdf.head())
    
    def train_init_from_file(self):
        # TODO: If training data has already been created we can load the pickle file
        pass

    def test_init(self):
        pass

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

x = pd.read_csv('data/train.csv', sep=';')

y = Model(x)

y.test()
y.train_init()

