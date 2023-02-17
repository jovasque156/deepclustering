#Data Handling
import numpy as np
import torch

#Pipelines
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack
from scipy import sparse
import scipy

#Datasets
from datasets.dataset import CustomDataset

#Plot
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
sns.set_theme()

#Transformation
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, Normalizer

#Imputation
#Do not remove enable_iterative_imputer, it is needed to import IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer

#Storing estimators
import pickle
import os

#Metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SELECTED = []

def load_data(data_path, train=True, test=True, merged=True):
    '''
    Load data from datasets/ folder
    Input:
        args: arguments from argparse
        train: load train data
        test: load test data
    
    Output:
        data_train: train data
        data_test: test data
    '''
    #Load data from datasets/ folder
    # checkpoint = torch.load(f"datasets/{args.data}/{args.data}.pt")
    checkpoint = torch.load(data_path)
    
    #Load train data
    if train:
        X_train, S_train, Y_train = checkpoint['train']
        data_train = CustomDataset(X_train, S_train, Y_train)

    #Load test data
    if test:
        X_test, S_test, Y_test = checkpoint['test']
        data_test = CustomDataset(X_test, S_test, Y_test)

    #Merge them in one
    if merged:
        X_train, S_train, Y_train = checkpoint['train']
        X_test, S_test, Y_test = checkpoint['test']

        X = sparse.vstack((X_train, X_test))
        S = np.concatenate((S_train, S_test))
        Y = np.concatenate((Y_train, Y_test))
        data = CustomDataset(X, S, Y)

    output = []
    if train: output.append(data_train)
    if test: output.append(data_test)
    if merged: output.append(data)
    
    return tuple(output)

def apply_preprocessing(X, nompipe, numpipe, pipe_normalize, idnumerical = None, idnominal = None):
    '''
    Apply transformer pipelines to X and returned the transformer dataset.

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
    Train transformer pipelines to X and returned the transformer dataset.

    Inputs:
    X: pandas (n,m), representing the dataset with n samples and m features to transform.
    idnumerical: numpy, representing the id of numerical features in X to transform using numpipe.
    idnominal: numpy, representing the id of nominal features in X to transform using nompipe.
    imputation: boolean, representing if imputation should be applied to X.
    encode: string, representing the type of encoding to apply to nominal features in X.
    standardscale: boolean, representing if standardization should be applied to X.
    normalize: boolean, representing if normalization should be applied to X.

    Outputs:
    X_final: csr_matrix, sparse matrix with the transformed X.
    pipe_nom: pipeline, representing the pipeline with transformer to apply to nominal features in X.
    pipe_num: pipeline, representing the pipeline with transformer to apply to numerical features in X.
    pipe_normalize: pipeline, representing the pipeline with transformer to apply to normalize the features in X.
    numerical: numpy, representing the name of numerical features
    nominal: numpy, representing the name of nominal features
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
            estimators.append(('imputation', imp_num))
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

def visualize(data, epoch, x, y, s, dec, args):
    '''
    Visualize the latent space of the model
    
    Inputs:
    data: string, representing the name of the dataset.
    epoch: int, representing the epoch of the model.
    x: tensor, representing the input data.
    y: tensor, representing the labels of the data.
    s: tensor, representing the sensitive attribute of the data.
    dec: object, representing the model to visualize.
    args: argparse, representing the arguments given.

    Outputs:
    Plot of the latent space of the model.
    
    '''

    # fig = plt.figure()
    # ax = plt.subplot(111)
    dec.eval()

    fig, axis = plt.subplots(1, 2)
    dec.to(DEVICE)
    dec.eval()

    q_ = dec(x.to(DEVICE))[0]
    cluster = q_.argmax(1).detach().cpu().numpy()
    x = dec.autoencoder.encode(x.to(DEVICE))
    x_embedded = x.detach().cpu().numpy()
    y = y.cpu().numpy()
    s = s.cpu().numpy()

    if x_embedded.shape[1]>2:
        # selected = random.sample(range(x.shape[0]), sampled)
        x_embedded = x_embedded[:args.sampled]
        y = y[:args.sampled]
        s = s[:args.sampled]
        cluster = cluster[:args.sampled]
        x_embedded = TSNE(n_components=2, learning_rate='auto', random_state=1).fit_transform(x_embedded)
    
        
    # plt.scatter(x_embedded[:,0], x_embedded[:,1], c=cluster, s=.5)
    sns.scatterplot(x=x_embedded[:,0], y=x_embedded[:,1], hue=cluster, s=1.2, ax=axis[0], palette=sns.color_palette())
    sns.scatterplot(x=x_embedded[:,0], y=x_embedded[:,1], hue=s, s=1.2, ax=axis[1], palette=sns.color_palette())
    
    axis[0].set_title('cluster')
    axis[1].set_title('sensitive')
    
    if not os.path.exists(f"plots/{data}/clusters{args.n_clusters}_gamma{args.gamma}_rho{args.rho}_beta{args.beta}_latentsize{args.latent_size_sae}_runid{args.run_id}"):
        os.makedirs(f"plots/{data}/clusters{args.n_clusters}_gamma{args.gamma}_rho{args.rho}_beta{args.beta}_latentsize{args.latent_size_sae}_runid{args.run_id}")

    fig.savefig(f'plots/{data}/clusters{args.n_clusters}_gamma{args.gamma}_rho{args.rho}_beta{args.beta}_latentsize{args.latent_size_sae}_runid{args.run_id}/ep_{epoch}_cluster.png')
    plt.close(fig)

    # fig = plt.figure()
    # ax = plt.subplot(111)
    # plt.scatter(x_embedded[:,0], x_embedded[:,1], c=s, s=.5)
    # fig.savefig(f'plots/{data}/clusters{num_clusters}_gamma{gamma}/ep_{epoch}_sensitive.png')
    # plt.close(fig)
    
def balance(S, assignment, n_clusters):
    '''
    Compute the balance of the clusters
    Inputs:
    S: numpy, representing the sensitive attribute of the data.
    assignment: numpy, representing the cluster assignment of the data.
    n_clusters: number of clusters.

    Outputs:
    balance: float, representing the balance of the clusters.
    '''
    # ipdb.set_trace()
    groups = np.unique(S, return_counts=False)

    balances_clusters = np.ones((n_clusters), dtype=float)

    for c in range(n_clusters):
        balances = np.ones((len(groups), len(groups)), dtype=float)
        for i in range(len(groups)):
            for j in range(len(groups)):
                if i != j:
                    balances[i, j] = np.sum(np.logical_and(assignment == c, S == groups[i]))/np.sum(np.logical_and(assignment == c, S == groups[j]))

        balances_clusters[c] = np.min(balances)

    return balances_clusters

def similarity_matrix(points):
    '''
    Compute similarity matrix using matrix

    Inputs:
    points: tensor, representing the data points.

    Outputs:
    sim_matrix: tensor, representing the similarity matrix.
    '''

    # ipdb.set_trace()
    matrix = torch.mm(points, points.t())
    diag = matrix.diag().unsqueeze(0)
    diag = diag.expand_as(matrix)

    sim = diag + diag.t() - 2*matrix

    return sim.sqrt()

# from https://github.com/enaserianhanzaei/DEC_Pytorch_tutorial/blob/master/utils/metrics.py
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

try:
    np.array(5).astype(float, copy=False)
except TypeError:
    # Compat where astype accepted no copy argument
    def astype(array, dtype, copy=True):
        if not copy and array.dtype == dtype:
            return array
        return array.astype(dtype)
else:
    astype = np.ndarray.astype

def linear_assignment(X):
    """Solve the linear assignment problem using the Hungarian algorithm.
    The problem is also known as maximum weight matching in bipartite graphs.
    The method is also known as the Munkres or Kuhn-Munkres algorithm.
    Parameters
    ----------
    X : array
        The cost matrix of the bipartite graph
    Returns
    -------
    indices : array,
        The pairs of (row, col) indices in the original array giving
        the original ordering.
    References
    ----------
    1. http://www.public.iastate.edu/~ddoty/HungarianAlgorithm.html
    2. Harold W. Kuhn. The Hungarian Method for the assignment problem.
       *Naval Research Logistics Quarterly*, 2:83-97, 1955.
    3. Harold W. Kuhn. Variants of the Hungarian method for assignment
       problems. *Naval Research Logistics Quarterly*, 3: 253-258, 1956.
    4. Munkres, J. Algorithms for the Assignment and Transportation Problems.
       *Journal of the Society of Industrial and Applied Mathematics*,
       5(1):32-38, March, 1957.
    5. http://en.wikipedia.org/wiki/Hungarian_algorithm
    """
    indices = _hungarian(X).tolist()
    indices.sort()
    # Re-force dtype to ints in case of empty list
    indices = np.array(indices, dtype=int)
    # Make sure the array is 2D with 2 columns.
    # This is needed when dealing with an empty list
    indices.shape = (-1, 2)
    return indices

class _HungarianState(object):
    """State of one execution of the Hungarian algorithm.
    Parameters
    ----------
    cost_matrix : 2D matrix
        The cost matrix. Does not have to be square.
    """

    def __init__(self, cost_matrix):
        cost_matrix = np.atleast_2d(cost_matrix)

        # If there are more rows (n) than columns (m), then the algorithm
        # will not be able to work correctly. Therefore, we
        # transpose the cost function when needed. Just have to
        # remember to swap the result columns back later.
        transposed = (cost_matrix.shape[1] < cost_matrix.shape[0])
        if transposed:
            self.C = (cost_matrix.T).copy()
        else:
            self.C = cost_matrix.copy()
        self.transposed = transposed

        # At this point, m >= n.
        n, m = self.C.shape
        self.row_uncovered = np.ones(n, dtype=np.bool)
        self.col_uncovered = np.ones(m, dtype=np.bool)
        self.Z0_r = 0
        self.Z0_c = 0
        self.path = np.zeros((n + m, 2), dtype=int)
        self.marked = np.zeros((n, m), dtype=int)

    def _find_prime_in_row(self, row):
        """
        Find the first prime element in the specified row. Returns
        the column index, or -1 if no starred element was found.
        """
        col = np.argmax(self.marked[row] == 2)
        if self.marked[row, col] != 2:
            col = -1
        return col

    def _clear_covers(self):
        """Clear all covered matrix cells"""
        self.row_uncovered[:] = True
        self.col_uncovered[:] = True


def _hungarian(cost_matrix):
    """The Hungarian algorithm.
    Calculate the Munkres solution to the classical assignment problem and
    return the indices for the lowest-cost pairings.
    Parameters
    ----------
    cost_matrix : 2D matrix
        The cost matrix. Does not have to be square.
    Returns
    -------
    indices : 2D array of indices
        The pairs of (row, col) indices in the original array giving
        the original ordering.
    """
    state = _HungarianState(cost_matrix)

    # No need to bother with assignments if one of the dimensions
    # of the cost matrix is zero-length.
    step = None if 0 in cost_matrix.shape else _step1

    while step is not None:
        step = step(state)

    # Look for the starred columns
    results = np.array(np.where(state.marked == 1)).T

    # We need to swap the columns because we originally
    # did a transpose on the input cost matrix.
    if state.transposed:
        results = results[:, ::-1]

    return results

# Individual steps of the algorithm follow, as a state machine: they return
# the next step to be taken (function to be called), if any.

def _step1(state):
    """Steps 1 and 2 in the Wikipedia page."""

    # Step1: For each row of the matrix, find the smallest element and
    # subtract it from every element in its row.
    state.C -= state.C.min(axis=1)[:, np.newaxis]
    # Step2: Find a zero (Z) in the resulting matrix. If there is no
    # starred zero in its row or column, star Z. Repeat for each element
    # in the matrix.
    for i, j in zip(*np.where(state.C == 0)):
        if state.col_uncovered[j] and state.row_uncovered[i]:
            state.marked[i, j] = 1
            state.col_uncovered[j] = False
            state.row_uncovered[i] = False

    state._clear_covers()
    return _step3


def _step3(state):
    """
    Cover each column containing a starred zero. If n columns are covered,
    the starred zeros describe a complete set of unique assignments.
    In this case, Go to DONE, otherwise, Go to Step 4.
    """
    marked = (state.marked == 1)
    state.col_uncovered[np.any(marked, axis=0)] = False

    if marked.sum() < state.C.shape[0]:
        return _step4


def _step4(state):
    """
    Find a noncovered zero and prime it. If there is no starred zero
    in the row containing this primed zero, Go to Step 5. Otherwise,
    cover this row and uncover the column containing the starred
    zero. Continue in this manner until there are no uncovered zeros
    left. Save the smallest uncovered value and Go to Step 6.
    """
    # We convert to int as numpy operations are faster on int
    C = (state.C == 0).astype(np.int)
    covered_C = C * state.row_uncovered[:, np.newaxis]
    covered_C *= astype(state.col_uncovered, dtype=np.int, copy=False)
    n = state.C.shape[0]
    m = state.C.shape[1]
    while True:
        # Find an uncovered zero
        row, col = np.unravel_index(np.argmax(covered_C), (n, m))
        if covered_C[row, col] == 0:
            return _step6
        else:
            state.marked[row, col] = 2
            # Find the first starred element in the row
            star_col = np.argmax(state.marked[row] == 1)
            if not state.marked[row, star_col] == 1:
                # Could not find one
                state.Z0_r = row
                state.Z0_c = col
                return _step5
            else:
                col = star_col
                state.row_uncovered[row] = False
                state.col_uncovered[col] = True
                covered_C[:, col] = C[:, col] * (
                    astype(state.row_uncovered, dtype=np.int, copy=False))
                covered_C[row] = 0


def _step5(state):
    """
    Construct a series of alternating primed and starred zeros as follows.
    Let Z0 represent the uncovered primed zero found in Step 4.
    Let Z1 denote the starred zero in the column of Z0 (if any).
    Let Z2 denote the primed zero in the row of Z1 (there will always be one).
    Continue until the series terminates at a primed zero that has no starred
    zero in its column. Unstar each starred zero of the series, star each
    primed zero of the series, erase all primes and uncover every line in the
    matrix. Return to Step 3
    """
    count = 0
    path = state.path
    path[count, 0] = state.Z0_r
    path[count, 1] = state.Z0_c

    while True:
        # Find the first starred element in the col defined by
        # the path.
        row = np.argmax(state.marked[:, path[count, 1]] == 1)
        if not state.marked[row, path[count, 1]] == 1:
            # Could not find one
            break
        else:
            count += 1
            path[count, 0] = row
            path[count, 1] = path[count - 1, 1]

        # Find the first prime element in the row defined by the
        # first path step
        col = np.argmax(state.marked[path[count, 0]] == 2)
        if state.marked[row, col] != 2:
            col = -1
        count += 1
        path[count, 0] = path[count - 1, 0]
        path[count, 1] = col

    # Convert paths
    for i in range(count + 1):
        if state.marked[path[i, 0], path[i, 1]] == 1:
            state.marked[path[i, 0], path[i, 1]] = 0
        else:
            state.marked[path[i, 0], path[i, 1]] = 1

    state._clear_covers()
    # Erase all prime markings
    state.marked[state.marked == 2] = 0
    return _step3


def _step6(state):
    """
    Add the value found in Step 4 to every element of each covered row,
    and subtract it from every element of each uncovered column.
    Return to Step 4 without altering any stars, primes, or covered lines.
    """
    # the smallest uncovered value in the matrix
    if np.any(state.row_uncovered) and np.any(state.col_uncovered):
        minval = np.min(state.C[state.row_uncovered], axis=0)
        minval = np.min(minval[state.col_uncovered])
        state.C[np.logical_not(state.row_uncovered)] += minval
        state.C[:, state.col_uncovered] -= minval
    return _step4