import torch
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

#Pipelines
from utils import apply_preprocessing, train_preprocessing

#Debugging
import ipdb

#TODO: add the preprocess_datasetes.py from the FUFDP project. There are more datasets.

DIR_DATA = {
    'census_income':'datasets/census_income/',
    'compas': 'datasets/compas/',
    'dutch_census': 'datasets/dutch_census/',
    'german_data': 'datasets/german_credit_data/'
    }

DIR_DATA_TRAIN = {
        'census_income':'adult.data',
        'compas': 'compas.csv',
        'dutch_census': 'dutch_census.csv',
        'german_data': 'german_data.csv'
    }

DIR_DATA_TEST = {
        'census_income':'adult.test',
        'compas': None,
        'dutch_census': None,
        'german_data': None
    }

FEATURES = {
    'census_income': ['age',
                    'workclass',
                    'education', 
                    'education-num', 
                    'marital-status', 
                    'occupation', 
                    'relationship', 
                    'race',
                    'sex', 
                    'capital-gain', 
                    'capital-loss', 
                    'hours-per-week',
                    'native-country'],
    'compas': ['sex',
            'age',
            'age_cat',
            'race',
            'juv_fel_count',
            'juv_misd_count',
            'juv_other_count',
            'priors_count',
            'c_days_jail',
            'c_charge_degree'],
    'dutch_census': ['sex', 
                    'age',
                    'household_position',
                    'household_size',
                    'prev_residence_place',
                    'citizenship',
                    'country_birth',
                    'edu_level',
                    'economic_status',
                    'cur_eco_activity',
                    'Marital_status'],
    'german_data': ['status_existing_check_account',
                    'duration_month', 
                    'credit_history',
                    'purpose',
                    'credit_amount',
                    'saving_account_bonds',	
                    'present_employe_since',
                    'installment_rate',
                    'personal_status_sex',
                    'sex', 
                    'other_debtors', 
                    'present_residence_since',
                    'property',
                    'age_years',
                    'other_installment', 
                    'housing', 
                    'num_existing_credits',
                    'job',
                    'num_people_liable',
                    'telephone',
                    'foreign_worker']
}


NOMINAL = {
    'census_income': ['workclass', 
                    'education', 
                    'marital-status', 
                    'occupation', 
                    'relationship', 
                    'race', 
                    'native-country'],
    'compas': ['age_cat',
                'sex',
                'race',
                'c_charge_degree'],
    'dutch_census': ['sex',
                    'age',
                    'household_position',
                    'household_size',
                    'prev_residence_place',
                    'citizenship',
                    'country_birth',
                    'edu_level',
                    'economic_status',
                    'cur_eco_activity',
                    'Marital_status'],
    'german_data': ['status_existing_check_account',
                    'credit_history',
                    'purpose',
                    'saving_account_bonds',	
                    'present_employe_since',
                    'personal_status_sex',
                    'sex',
                    'other_debtors',
                    'property',
                    'other_installment', 
                    'housing', 
                    'job',
                    'telephone',
                    'foreign_worker']
    }


#The first is the name of the attribute, and the second is the groups
#The list of the groups should start with the protected group.
SENSITIVE_ATTRIBUTE = {
    'census_income': {'sex': ['Female', 'Male']},
    'compas': {'race': ['African-American', 'Caucasian', 'Hispanic', 'Native American', 'Other']},
    'dutch_census': {'sex': ['Female', 'Male']},
    'german_data': {'sex': ['A01', 'A02']} #TODO: add the age after binarization young/old <=25/>25.
    
    }

LABEL = {
    'census_income': {'income': ['>50K', '<=50K']},
    'compas': {'two_year_recid': [1, 0], 'is_recid': [1, 0]},
    'dutch_census': {'occupation': ['high_level', 'low-level']},
    'german_data': {'class': [1, 2]}
}

DEFAULT_SENSITIVE_ATTRIBUTE= {
    'census_income': 'sex',
    'compas': 'race',
    'dutch_census': 'sex',
    'german_data': 'sex' #TODO: add the age after binarization young/old <=25/>25.
}

DEFAULT_LABEL = {
    'census_income': 'income',
    'compas': 'two_year_recid',
    'dutch_census': 'occupation',
    'german_data': 'class'
}

def preprocess_datasets(args):
    '''
    Preprocess the datasets and save them in the datasets/ folder

    Output:
        - X_train: sparse matrix, representing the features
        - S_train: numpy, representing the sensitive attribute. Assuming binary
        - Y_train: numpy, representing the label.
    '''
    # Load the data
    if None in [DIR_DATA_TEST[args.dataset], DIR_DATA_TRAIN[args.dataset]]:
        df = pd.read_csv(DIR_DATA[args.dataset]+DIR_DATA_TRAIN[args.dataset])    
        df_train, df_test = train_test_split(df, test_size=args.test_size)
    else:
        df_train = pd.read_csv(DIR_DATA[args.dataset]+DIR_DATA_TRAIN[args.dataset])
        df_test = pd.read_csv(DIR_DATA[args.dataset]+DIR_DATA_TEST[args.dataset])
    
    #We are assuming binary target variable and sensitive attribute. For sensitive attribute, the first is the group 1 and the rest the 0.
    #TODO: handle multi-class in target and sensitive attribute
    features = FEATURES[args.dataset]
    features.remove(args.sensitive_attribute)

    # Retrieve variables
    Y_train = df_train.loc[:, [args.target_variable]].to_numpy().flatten()
    Y_train = 1*(Y_train == LABEL[args.dataset][args.target_variable][0])
    S_train = df_train.loc[:, [args.sensitive_attribute]].to_numpy().flatten()
    S_train = 1*(S_train == SENSITIVE_ATTRIBUTE[args.dataset][args.sensitive_attribute][0])
    X_train = df_train.loc[:, features]
    
    Y_test = df_test.loc[:, [args.target_variable]].to_numpy().flatten()
    Y_test = 1*(Y_test == LABEL[args.dataset][args.target_variable][0])
    S_test = df_test.loc[:, [args.sensitive_attribute]].to_numpy().flatten()
    S_test = 1*(S_test == SENSITIVE_ATTRIBUTE[args.dataset][args.sensitive_attribute][0])
    X_test = df_test.loc[:, features]

    # Get nominal names of features and remove the sensitive attribute
    nominal_names = NOMINAL[args.dataset]
    if args.sensitive_attribute in nominal_names: nominal_names.remove(args.sensitive_attribute)

    # Get id_numerical
    id_numerical = [i 
                    for i, f in enumerate(X_train.columns)
                    if f not in nominal_names]

    # Encode the categorical features
    (outcome) = train_preprocessing(X_train, 
                                    idnumerical=id_numerical, 
                                    imputation=args.not_imputation, 
                                    encode=args.nominal_encode, 
                                    standardscale=args.standardscale,
                                    normalize = args.normalize)
    
    X_train, pipe_num, pipe_nom, pipe_normalize, numerical_features, nominal_features = outcome
    
    X_test = apply_preprocessing(X_test, 
                                pipe_nom, 
                                pipe_num, 
                                pipe_normalize, 
                                idnumerical=id_numerical)

    torch.save({
            'train': (X_train, S_train, Y_train),
            'test': (X_test, S_test, Y_test),
            'pipes': (pipe_nom, pipe_num),
            'features': (numerical_features, nominal_features),
            }, DIR_DATA[args.dataset]+args.dataset+'.pt')

    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='census_income', help='Dataset to preprocess')
    parser.add_argument('--test_size', type=float, default=0.3, help='Size of test if is not defined')
    parser.add_argument('--sensitive_attribute', type=str, default='sex', help='features used as sensitive attribute')
    parser.add_argument('--target_variable', type=str, default='target', help='target variable')
    parser.add_argument('--nominal_encode', type=str, default='label', help='Type of encoding for nominal features')
    parser.add_argument('--standardscale', action="store_true", help='Apply standard scale transformation')
    parser.add_argument('--normalize', action="store_true", help='Apply normalization transformation')
    parser.add_argument('--not_imputation', action="store_false", help='Set false to not apply imputation on missing values')
    
    #TODO: use the following two parameters to handle multi-class targets and sensitive attributes
    parser.add_argument('--target_multi_class', action='store_true', help='target variable as multi-class')
    parser.add_argument('--sa_multi_class', action='store_true', help='sensitive attribute as multi-class')
    
    args = parser.parse_args()

    if args.dataset == 'all':
        for ds in DIR_DATA.keys():
            args.dataset = ds
            args.sensitive_attribute = DEFAULT_SENSITIVE_ATTRIBUTE[ds]
            args.target_variable = DEFAULT_LABEL[ds]

            print(f'Preprocessing {args.dataset} dataset...')
            preprocess_datasets(args)
            print('Done!')
    else:
        print(f'Preprocessing {args.dataset} dataset...')
        preprocess_datasets(args)
        print('Done!')