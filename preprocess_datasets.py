import torch
import numpy as np
import pandas as pd
import argparse

#Pipelines
from utils import apply_preprocessing, train_preprocessing

#Debugging
import ipdb

DIR_DATA = {
    'census_income':'datasets/census_income/'
}

DIR_DATA_TRAIN = {
        'census_income':'adult.data'
    }

DIR_DATA_TEST = {
        'census_income':'adult.test'
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
                    'native-country']
}

NOMINAL = {
    'census_income': ['workclass', 
                    'education', 
                    'marital-status', 
                    'occupation', 
                    'relationship', 
                    'race', 
                    'sex',
                    'native-country']}


#The first is the name of the attribute, and the second is the groups
#The list of the groups should start with the protected group.
SENSITIVE_ATTRIBUTE = {
    'census_income': ('sex', ['Female', 'Male'])
    }

LABEL = {
    'census_income': ('income', ['>50K', '<=50K'])
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
    df_train = pd.read_csv(DIR_DATA[args.dataset]+DIR_DATA_TRAIN[args.dataset])
    df_test = pd.read_csv(DIR_DATA[args.dataset]+DIR_DATA_TEST[args.dataset])

    # Drop the 'fnlwgt' feature
    if args.dataset == 'census_income':
        df_train = df_train.drop('fnlwgt', axis=1)
        df_test = df_test.drop('fnlwgt', axis=1)
    
    # Retrieve variables
    Y_train = df_train.loc[:, [LABEL[args.dataset][0]]].to_numpy().flatten()
    # This is not totally correct, since it might be multi-class, but it works for the moment
    Y_train = 1*(Y_train == LABEL[args.dataset][1][0])
    S_train = df_train.loc[:, [SENSITIVE_ATTRIBUTE[args.dataset][0]]].to_numpy().flatten()
    S_train = 1*(S_train == SENSITIVE_ATTRIBUTE[args.dataset][1][0])
    X_train = df_train.loc[:, FEATURES[args.dataset]]
    
    Y_test = df_test.loc[:, [LABEL[args.dataset][0]]].to_numpy().flatten()
    # This is not totally correct, since it might be multi-class, but it works for the moment
    Y_test = 1*(Y_test == LABEL[args.dataset][1][0])
    S_test = df_test.loc[:, [SENSITIVE_ATTRIBUTE[args.dataset][0]]].to_numpy().flatten()
    S_test = 1*(S_test == SENSITIVE_ATTRIBUTE[args.dataset][1][0])
    X_test = df_test.loc[:, FEATURES[args.dataset]]

    # Get id_numerical
    id_numerical = [i 
                    for i, f in enumerate(X_train.columns)
                    if f not in NOMINAL[args.dataset]]

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
            }, DIR_DATA[args.dataset]+'census_income.pt')

    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='census_income', help='Dataset to preprocess')
    parser.add_argument('--nominal_encode', type=str, default='label', help='Type of encoding for nominal features')
    parser.add_argument('--standardscale', action="store_true", help='Apply standard scale transformation')
    parser.add_argument('--normalize', action="store_true", help='Apply normalization transformation')
    parser.add_argument('--not_imputation', action="store_false", help='Set false to not apply imputation on missing values')
    
    args = parser.parse_args()

    print(f'Preprocessing {args.dataset} dataset...')
    preprocess_datasets(args)
    print('Done!')