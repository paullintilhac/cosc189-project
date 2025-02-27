"""
Utility file containing helper functions that perform various dimensionality
reduction technique.
"""
import numpy as np

from sklearn.decomposition import PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection as GRP

from lib.utils.data_utils import get_data_shape
from lib.utils.DCA import DCA
from lib.utils.AntiWhiten import AntiWhiten

#------------------------------------------------------------------------------#


def kernel_pca_dr(X_train, X_test, rd,kernel="linear",gamma=None,X_val=None, rev=None, **kwargs):
    """
    Perform kernel PCA on X_train then transform X_train, X_test (and X_val).
    Return transformed data in original space if rev is True; otherwise, return
    transformed data in PCA space.
    """

    print("running kernel pca dr with kernel " + kernel)
    whiten = kwargs['whiten']
    # Fit PCA model on training data, random_state is specified to make sure
    # result is reproducible
    kpca = KernelPCA(n_components=rd, fit_inverse_transform=rev,kernel=kernel, gamma=gamma, random_state=10)
    print("kpca: " + str(kpca))

    kpca.fit(X_train)

    # Transforming training and test data
    X_train_dr = kpca.transform(X_train)
    X_test_dr = kpca.transform(X_test)
    if X_val is not None:
        X_val_dr = kpca.transform(X_val)

    if rev is not None:
        X_train_rev = kpca.inverse_transform(X_train_dr)
        X_test_rev = kpca.inverse_transform(X_test_dr)
        if X_val is not None:
            X_val_rev = kpca.inverse_transform(X_val_dr)
            return X_train_rev, X_test_rev, X_val_rev, kpca
        else:
            return X_train_rev, X_test_rev, kpca
    else:
        if X_val is not None:
            print("dr_alg: " + str(kpca))
            return X_train_dr, X_test_dr, X_val_dr, kpca
        else:
            print("dr_alg when X_val is none: " + str(kpca))

            return X_train_dr, X_test_dr, kpca
#------------------------------------------------------------------------------#

def pca_dr(X_train, X_test, rd, X_val=None, rev=None, **kwargs):
    """
    Perform PCA on X_train then transform X_train, X_test (and X_val).
    Return transformed data in original space if rev is True; otherwise, return
    transformed data in PCA space.
    """
    print("running pca dr")
    whiten = kwargs['whiten']
    # Fit PCA model on training data, random_state is specified to make sure
    # result is reproducible
    pca = PCA(n_components=rd, whiten=whiten, random_state=10)
    print("pca: " + str(pca))

    pca.fit(X_train)

    # Transforming training and test data
    X_train_dr = pca.transform(X_train)
    X_test_dr = pca.transform(X_test)
    if X_val is not None:
        X_val_dr = pca.transform(X_val)

    if rev is not None:
        X_train_rev = pca.inverse_transform(X_train_dr)
        X_test_rev = pca.inverse_transform(X_test_dr)
        if X_val is not None:
            X_val_rev = pca.inverse_transform(X_val_dr)
            return X_train_rev, X_test_rev, X_val_rev, pca
        else:
            return X_train_rev, X_test_rev, pca
    else:
        if X_val is not None:
            return X_train_dr, X_test_dr, X_val_dr, pca
        else:
            return X_train_dr, X_test_dr, pca
#------------------------------------------------------------------------------#


def random_proj_dr(X_train, X_test, rd, X_val=None, rev=None, **kwargs):
    """
    Perform Gaussian Random Projection on X_train then transform X_train, X_test
    (and X_val). Return transformed data in original space if rev is True;
    otherwise, return transformed data in PCA space.
    """

    # Generating random matrix for projection
    grp = GRP(n_components=rd, random_state=10)
    X_train_dr = grp.fit_transform(PCA_in_train)
    X_test_dr = grp.transform(PCA_in_test)

    X_train_dr = X_train_dr.reshape((train_len, 1, rd))
    X_test_dr = X_test_dr.reshape((test_len, 1, rd))

    return X_train_dr, X_test_dr, grp
#------------------------------------------------------------------------------#


def dca_dr(X_train, X_test, rd, X_val=None, rev=None, **kwargs):
    """
    Perform DCA on X_train then transform X_train, X_test (and X_val).
    Return transformed data in original space if rev is True; otherwise, return
    transformed data in DCA space.
    """

    y_train = kwargs['y_train']
    if y_train is None:
        raise ValueError('y_train is required for DCA')

    # Fit DCA model on training data, random_state is specified to make sure
    # result is reproducible
    dca = DCA(n_components=rd, rho=None, rho_p=None)
    dca.fit(X_train, y_train)

    # Transforming training and test data
    X_train_dr = dca.transform(X_train)
    X_test_dr = dca.transform(X_test)
    if X_val is not None:
        X_val_dr = dca.transform(X_val)

    if rev is not None:
        X_train_rev = dca.inverse_transform(X_train_dr)
        X_test_rev = dca.inverse_transform(X_test_dr)
        if X_val is not None:
            X_val_rev = dca.inverse_transform(X_val_dr)
            return X_train_rev, X_test_rev, X_val_rev, dca
        else:
            return X_train_rev, X_test_rev, dca
    else:
        if X_val is not None:
            return X_train_dr, X_test_dr, X_val_dr, dca
        else:
            return X_train_dr, X_test_dr, dca
#------------------------------------------------------------------------------#


def anti_whiten_dr(X_train, X_test, rd, X_val=None, rev=None, **kwargs):
    """
    Perform dimensionality reduction with eigen-based whitening preprocessing
    """

    deg = kwargs['deg']
    # Fit X_train
    anti_whiten = AntiWhiten(n_components=rd, deg=deg)
    anti_whiten.fit(X_train)

    # Transform X_train and X_test
    X_train_dr = anti_whiten.transform(X_train)
    X_test_dr = anti_whiten.transform(X_test)
    if X_val:
        # Transform X_Val
        X_val_dr = anti_whiten.transform(X_val)

    if rev:
        X_train_rev = anti_whiten.inverse_transform(X_train_dr, inv_option=1)
        X_test_rev = anti_whiten.inverse_transform(X_test_dr, inv_option=1)
        if X_val:
            X_val_rev = anti_whiten.inverse_transform(X_val_dr, inv_option=1)
            return X_train_rev, X_test_rev, X_val_rev, anti_whiten
        else:
            return X_train_rev, X_test_rev, anti_whiten
    else:
        if X_val:
            return X_train_dr, X_test_dr, X_val_dr, anti_whiten
        else:
            return X_train_dr, X_test_dr, anti_whiten
#------------------------------------------------------------------------------#


def invert_dr(X, dr_alg, DR):
    """
    Inverse transform data <X> in reduced dimension back to its full dimension
    """

    inv_list = ['pca', 'pca-whiten', 'dca','kernel-pca']

    if (DR in inv_list) or ('antiwhiten' in DR):
        X_rev = dr_alg.inverse_transform(X)
    else:
        raise ValueError('Cannot invert specified DR')

    return X_rev
#------------------------------------------------------------------------------#


def gradient_transform(model_dict, dr_alg):

    DR = model_dict['dim_red']
    rev = model_dict['rev']

    # A is transformation matrix of dr_alg
    if DR == 'pca' or DR == 'kernel-pca':
        #print("doing gradient transform")
        if rev == None:
            A = dr_alg.components_
        elif rev != None:
            B = dr_alg.components_
            A = np.dot(B.T, B)
    elif DR == 'pca-whiten':
        # This S is S / sqrt(n_samples)
        # Entries in S with very small value ~0 (last few elements) could cause
        # stability problem when inverted
        S_inv = 1 / np.sqrt(dr_alg.explained_variance_)
        V = dr_alg.components_.T
        # A = (V / S).T
        A = np.dot(V, np.diag(S_inv)).T
    elif 'antiwhiten' in DR:
        deg = int(DR.split('antiwhiten', 1)[1])
        S = dr_alg.S_
        if deg == -1:
            A = np.diag(1 / S)
        elif deg == 0:
            A = np.eye(dr_alg.n_components)
        elif deg >= 1:
            A = np.eye(dr_alg.n_components)
            for i in range(deg):
                A = np.dot(A, np.diag(S))
        A = np.dot(dr_alg.V_, A).T
    else:
        raise ValueError('Cannot get transformation matrix from this \
                          dimensionality reduction')
    #print("gradient transform for kernel-pca")
    #print("A: "+str(A))
    return A
#------------------------------------------------------------------------------#


def dr_wrapper(X_train, X_test, X_val, DR, rd, y_train, rev=None,small=None,gamma=None,kernel='linear'):
    """
    A wrapper function for dimensionality reduction functions.
    """

    data_dict = get_data_shape(X_train, X_test, X_val)
    no_of_dim = data_dict['no_of_dim']

    train_len = data_dict['train_len']
    test_len = data_dict['test_len']
    no_of_features = data_dict['no_of_features']

    # Reshape for dimension reduction function
    DR_in_train = X_train.reshape(train_len, no_of_features)
    DR_in_test = X_test.reshape(test_len, no_of_features)
    if X_val.any():
        val_len = data_dict['val_len']
        DR_in_val = X_val.reshape(val_len, no_of_features)
    else:
        DR_in_val = None

    whiten = None
    deg = None

    # Assign corresponding DR function
    if DR == 'pca':
        dr_func = pca_dr
        whiten = False
    if DR == 'pca-whiten':
        dr_func = pca_dr
        whiten = True
    if DR == 'kernel-pca':
        dr_func = kernel_pca_dr
        whiten = False
    elif DR == 'rp':
        dr_func = random_proj_dr
    elif DR == 'dca':
        dr_func = dca_dr
    elif 'antiwhiten' in DR:
        dr_func = anti_whiten_dr
        deg = int(DR.split('antiwhiten', 1)[1])

    # Perform DR
    if X_val.any():
        X_train, X_test, X_val, dr_alg = dr_func(DR_in_train, DR_in_test, rd,
                                                X_val=DR_in_val, rev=rev,
                                                y_train=y_train, whiten=whiten,

                                                deg=deg,small=small,gamma=gamma,kernel=kernel)
    else:
        X_train, X_test, dr_alg = dr_func(DR_in_train, DR_in_test, rd,
                                          X_val=DR_in_val, rev=rev,
                                          y_train=y_train, whiten=whiten,
                                          deg=deg,small=small,gamma=gamma,kernel=kernel)

    # Reshape DR data to appropriate shape (original shape if rev)
    if (no_of_dim == 3) or ((no_of_dim == 4) and (rev is None)):
        channels = data_dict['channels']
        X_train = X_train.reshape((train_len, channels, rd))
        X_test = X_test.reshape((test_len, channels, rd))
        if X_val is not None and len(X_val)>0:
            X_val = X_val.reshape((val_len, channels, rd))
    elif (no_of_dim == 4) and (rev is not None):
        channels = data_dict['channels']
        height = data_dict['height']
        width = data_dict['width']
        X_train = X_train.reshape((train_len, channels, height, width))
        X_test = X_test.reshape((test_len, channels, height, width))
        if X_val is not None and len(X_val)>0:
            X_val = X_val.reshape((val_len, channels, height, width))
    print("dr_alg at the end off dr_wrapper: " + str(dr_alg))
    return X_train, X_test, X_val, dr_alg
#------------------------------------------------------------------------------#
