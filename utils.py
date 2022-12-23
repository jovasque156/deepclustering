#Data Handling
import pandas as pd
import numpy as np
import torch

#Pipelines
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from scipy import sparse
import scipy

#Plot
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

#Transformation
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, Normalizer

#Imputation
#Do not remove enable_iterative_imputer, it is needed to import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

#Storing estimators
import pickle
import os

#Debug
import ipdb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_preprocessing(X, nompipe, numpipe, pipe_normalize, idnumerical = None, idnominal = None):
    '''
    Apply tansformer pipelines to X and returned the transformer dataset.

    Inputs:
    X: pandas (n,m), representing the dataset with n samples and m features to transform.
    nompipe: pipeline, representing the pipeline with transformer to apply to nominal features in X.
    numpipe: pipeline, representing the pipeline with transformer to apply to numerical features in X.
    pipe_normalize: pipeline, representing the pipeline with transformer to apply to normalize the features in X.
    idnumerical: numpy, representing the id of numerical features in X to transform using numpipe.
    idnominal: numpy, representing the id of nominal features in X to transform using nompipe.

    Outputs:
    X_final: csr_matrix, sparse matrix with the transformed X. 
    '''
    if idnumerical==None:
        idnumerical = [i for i, e in enumerate(X.dtypes) if e == 'float64']
    if idnominal==None:
        idnominal = list(np.setdiff1d(list(range(0,len(X.columns))),idnumerical))
    
    numerical = list(X.iloc[:,idnumerical].columns)
    nominal = list(X.iloc[:,idnominal].columns)
    
    #Identifying numerical and nominal variables
    X_nom = X.loc[:,nominal]
    
    #Numerical
    X_num = X.loc[:,numerical]
    
    #Apply trained pipes
    if nompipe==None:
        X_num = numpipe.transform(X_num)
        X_final = X_num
    elif numpipe==None:
        X_nom = nompipe.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_final = X_nom
    else:
        X_nom = nompipe.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_num = numpipe.transform(X_num)
        X_final = hstack((X_num, X_nom))   
    
    if pipe_normalize!=None:
        X_final = pipe_normalize.transform(X_final)

    return X_final

def train_preprocessing(X, idnumerical=None, idnominal=None, imputation=True, encode = 'one-hot', standardscale=True, normalize = True ):
    '''
    Apply tansformer pipelines to X and returned the transformer dataset.

    Inputs:
    X: pandas (n,m), representing the dataset with n samples and m features to transform.
    idnumerical: numpy, representing the id of numerical features in X to transform using numpipe.
    idnominal: numpy, representing the id of nominal features in X to transform using nompipe.
    imputation: boolean, representing if missing values are imputed or not.
    encode: string, representing the type of encoding are applied to nominal features. It
                    can be one-hot or label.
    normalize: boolean, representing if numerical variables are normalized. 

    Outputs:
    X_final: csr_matrix, sparse matrix with the transformed X.
    '''
    #Identifying numerical and nominal variables
    if idnumerical==None:
        idnumerical = [i for i, e in enumerate(X.dtypes) if e == 'float64']
    if idnominal==None:
        idnominal = list(np.setdiff1d(list(range(0,len(X.columns))),idnumerical))
        
    numerical = list(X.iloc[:,idnumerical].columns)
    nominal = list(X.iloc[:,idnominal].columns)
        
    X_nom = X.loc[:,nominal]
    X_num = X.loc[:,numerical]
    

    #Applying estimators for nominal an numerical
    #nominal
    pipe_nom = None
    if len(nominal)>0:
        estimators = []
        if imputation == True:
            imp_nom = SimpleImputer(strategy='most_frequent')
            estimators.append(('imputation', imp_nom))
        if encode != None:
            enc = OneHotEncoder(drop='first') if encode=='one-hot' else OrdinalEncoder()
            estimators.append(('encoding', enc))
        if standardscale:
            scaler = StandardScaler(with_mean=True) if encode=='label' else StandardScaler(with_mean=False)
            estimators.append(('standardscale', scaler))

        pipe_nom = Pipeline(estimators)
        pipe_nom.fit(X_nom)
    
    #numerical
    pipe_num = None
    if len(numerical)>0:
        estimators = []
        if imputation == True:
            imp_num = IterativeImputer(max_iter=100, random_state=1)
            estimators.append(('impuation', imp_num))
        if standardscale:
            scale = StandardScaler(with_mean=True)
            estimators.append(('standardscale', scale))

        pipe_num = Pipeline(estimators)
        pipe_num.fit(X_num)
        
    #Merge both transformations
    if len(nominal)<1:
        X_num = pipe_num.transform(X_num)
        X_final = X_num
    elif len(numerical)<1:
        X_nom = pipe_nom.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_final = X_nom
    else:
        X_nom = pipe_nom.transform(X_nom)
        if not(scipy.sparse.issparse(X_nom)):
            X_nom = sparse.csr_matrix(X_nom)
        X_num = pipe_num.transform(X_num)
        X_final = hstack((X_num, X_nom))    
    
    # Add Normalize pipeline
    pipe_normalize = None
    if normalize:
        estimators = []
        estimators.append(('normalize', Normalizer()))
        pipe_normalize = Pipeline(estimators)
        pipe_normalize.fit(X_final)
        X_final = pipe_normalize.transform(X_final)

    hot_encoder = pipe_nom['encoding']
    if encode == 'one-hot':
        nom_features = hot_encoder.get_feature_names_out(nominal)
    else:
        nom_features = nominal
    

    return (X_final, pipe_num, pipe_nom , pipe_normalize, numerical, nom_features)

def import_pickle(directory):
    '''
    Load the pickle file in directory and return the corresponding object

    Input:
    directory: string, directory where pickle to load is located.

    Output:
    p: object, unpacked object from pickle in located at directory path.
    '''
    with open(directory, 'rb') as f:
        p = pickle.load(f)

    return p

def save_checkpoint(state, filename, is_best):
    '''
    Save the model checkpoint

    Inputs:
    state: dict, representing the model state to save.
    filename: string, representing the filename to save the model.
    is_best: boolean, representing if the model is the best or not.
    '''
    
    if is_best:
        print('=> Saving new checkpoint')
        torch.save(state, filename)
    
    else:
        print('=> Performance did not improve')

def visualize(data, epoch, x, y, s, dec, num_clusters, gamma):
    '''
    Visualize the latent space of the model
    Inputs:
    data: string, representing the name of the dataset.
    epoch: int, representing the epoch of the model.
    x: tensor, representing the input data.
    y: tensor, representing the labels of the data.
    s: tensor, representing the sensitive attribute of the data.
    dec: object, representing the model to visualize.
    num_clusters: int, representing the number of clusters of the model.

    Outputs:
    Plot of the latent space of the model.
    
    '''

    fig = plt.figure()
    ax = plt.subplot(111)
    dec.to(DEVICE)
    dec.eval()
    
    q_, phi_ = dec(x.to(DEVICE)) 
    cluster = q_.argmax(1).detach().cpu().numpy()[:4000]
    x = dec.autoencoder.encode(x.to(DEVICE))
    x = x.detach().cpu().numpy()[:4000]
    y = y.cpu().numpy()[:4000]
    s = s.cpu().numpy()[:4000]
    
    x_embedded = TSNE(n_components=2, learning_rate='auto', random_state=1).fit_transform(x)
    plt.scatter(x_embedded[:,0], x_embedded[:,1], c=cluster, s=.5)

    if not os.path.exists(f"plots/{data}/clusters{num_clusters}_gamma{gamma}"):
        os.makedirs(f"plots/{data}/clusters{num_clusters}_gamma{gamma}")

    fig.savefig(f'plots/{data}/clusters{num_clusters}_gamma{gamma}/ep_{epoch}_cluster.png')
    plt.close(fig)

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.scatter(x_embedded[:,0], x_embedded[:,1], c=s, s=.5)
    fig.savefig(f'plots/{data}/clusters{num_clusters}_gamma{gamma}/ep_{epoch}_sensitive.png')
    plt.close(fig)