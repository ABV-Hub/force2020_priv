import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost

# Supress chained pandas warnings
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def normalise(curve, ref_low, ref_high, well_low, well_high):
    
    norm = ref_low + ((ref_high - ref_low) * ((curve - well_low) / (well_high - well_low)))
    return norm

def bs_fix(bitsize):
    standard_bs_vals = [26, 17.5, 17, 14.75, 12.5, 12.25, 9.875, 8.5, 8.375, 6.5, 6]
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

def vol_shale(clean, shale, curve):
    vshale = (curve - clean) / (shale - clean)
    if vshale > 1:
        vshale = 1

    if vshale < 0:
        vshale = 0
    
    return vshale

def gardners_equation_rhob(curve):
    """Simplified Gardners Equation

    Parameters
    ----------
    curve :
        Compressional Slowness

    Returns
    -------
    float
        Computed RHOB from Gardner's Equation
    """
    a_metric = 0.31 #m/s
    a_imperial = 0.23 #ft/s
    b = 0.25
    
    vp = 1000000 / curve 
    rhob = a_imperial * (vp** b)

    return rhob

    # if target.lower()=='rhob' and dtc_units=='ft_s': # calculating RHOB from DTC
    #     vp = 1000000 / curve
    #     rhob = a_imperial* (vp**b)
    # elif target.lower()=='rhob' and dtc_units=='m_s':
    #     vp = 1000000 / curve
    #     rhob = a_metric* (vp**b)
    # elif target.lower()=='dtc' and dtc_units=='ft_s':
    #     vp = 1000000 / curve
    #     rhob = a_imperial* (vp**b)

def score(y_true, y_pred):
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]

def missingvals_plot(dataframe):
    pass


data = pd.read_csv('./data/train.csv', sep=';')

print(data.head())
columns = data.columns
A = np.load('penalty_matrix.npy')

# Convert lithology numbers

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

# Map the lithology numbers to new numbers between 0 and 11
data['FORCE_2020_LITHOFACIES_LITHOLOGY'] = data['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(lithology_numbers)

print(columns)

# Create our working data frame. A large number of the curves have significant null values and won't be easy to repair.
# Initially we will drop these.
working = data[['WELL', 'DEPTH_MD', 'CALI', 'RDEP', 'RMED', 'DRHO', 'GR', 'RHOB', 'NPHI', 'PEF', 'DTC',
 'SP', 'BS', 'Z_LOC', 'X_LOC', 'Y_LOC', 'FORCE_2020_LITHOFACIES_CONFIDENCE', 'FORCE_2020_LITHOFACIES_LITHOLOGY']]
 

 
# Normalise GR data

percentile_95 = working.groupby('WELL')['GR'].quantile(0.95)
working['95_PERC'] = working['WELL'].map(percentile_95)
percentile_05 = working.groupby('WELL')['GR'].quantile(0.05)
working['05_PERC'] = working['WELL'].map(percentile_05)

# Key Well High and Low
# 35/9-5 shows a nice distribution for shale and sand
print(f'Normalising Gamma Ray.....')
key_well_low = 50.822391
key_well_high = 131.688494
working['GR_NORM'] = working.apply(lambda x: normalise(x['GR'], key_well_low, key_well_high, x['05_PERC'], x['95_PERC']), axis=1)

print(f'Normalising Gamma Ray Complete....')
print(f'Calculating VShale......')

working['VSHALE'] = working.apply(lambda x: vol_shale(50.822391, 131.688494, x['GR_NORM']), axis=1)
print(f'Calculating VShale Complete......')

# Fixing bitsize due to "odd" bitsize values, some look interpolated others don't make sense
print('Fixing Existing Bitsize (BS) Curve.....')
working['BS_FIX'] = working.apply(lambda x: bs_fix(x.loc['BS']), axis=1)

# This is an attempt to create a synthetic bitsize curve based on a rolling average of the CALI curve
print('Creating Synthetic Bitsize From CALI.....')
working['CALI_R_MEAN'] = working['CALI'].rolling(10).mean()
working['BS_FROM_CALI'] = working.apply(lambda x: bs_fix(x.loc['CALI']), axis=1)

# Combine the Bitsize curves
print('Combining Existing Bitsize and Fixed Bitsize Curves')
working['BS_COMB']=working['BS_FIX']
working['BS_COMB'].fillna(working['BS_FROM_CALI'], inplace=True)

# Create a new differential caliper. Existing one had poor coverage
print('Calculating Diff CAL')
working['DIFF_CAL']=working['CALI'] - working['BS_COMB']

#Calculate a bad hole flag, where caliper is greater than 2" over Bitsize
# TODO: Introduce DRHO into the Bad Hole Calculation
print('Calculating Bad Hole Flag.....')
working['BADHOLE_CAL'] = np.where(working['DIFF_CAL']>=2, 1, 0)
working['BADHOLE_CAL'].value_counts().plot(kind='bar')
plt.show()

# Fix missing values in Density and DTC based on Gardners Equation


# If gaps are still present then fill in with the trend curves
print('Creating RHOB and DTC Trend Curves.....')
working['TVD'] = working['Z_LOC'] * -1
working['RHOB_GARD'] = working.apply(lambda x: gardners_equation_rhob(x.loc['DTC']), axis=1)
working['RHOB_FIX'] = working['RHOB']
working['RHOB_FIX'].fillna(working['RHOB_GARD'], inplace=True) # Not convinced this is working yet

plt.scatter(working['RHOB'], working['RHOB_FIX'], color='red', marker='.')
plt.xlim(1.5, 3)
plt.ylim(1.5 ,3)
plt.show()

# One Hot Encoder for Group?

print(working.head())


# Create initial training subset
training_data = working[['WELL', 'TVD', 'BADHOLE_CAL', 'RDEP', 'VSHALE', 'RHOB_FIX', 'DTC', 'FORCE_2020_LITHOFACIES_LITHOLOGY']]
training_data.to_pickle('initial_training_data')


# Load Data
# Create Subset - DEPT, CALI, RDEP, RMED, DRHO, GR, NPHI, PEF, DTC, SP, BS Z_LOC, X_LOC, Y_LOC, Formation
# And LITH Curves

# Create Bad Hole flags from CALI & DRHO
# Compute missing RHOB values using Gardeners Eq
# Compute missin DTC from RHOB using Gardeners Eq

class Model_Build:
    def __init__(self, dataframe):
        self.data = dataframe
        print('Dataframe sucessfully loaded')
        print(self.data.head())
        self.get_key_features()
    
    def get_key_features(self):
        print(self.data.columns)