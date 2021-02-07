import numpy as np
import pandas as pd
import random



## all samples MUST include at least one view.
def import_incomplete_handwritten():
    npz = np.load('./data/Handwritten_Missing/data_with_missingviews.npz', allow_pickle=True)

    X_set    = npz['X_set'].tolist()
    Y_onehot = npz['Y_onehot']

    M        = len(X_set)

    ### Construct Mask Vector to indicate available (m=1) or missing (m=0) values
    Mask     = np.ones([np.shape(X_set[0])[0], M])
    for m_idx in range(M):
        Mask[np.isnan(X_set[m_idx]).all(axis=1), m_idx] = 0
        X_set[m_idx][Mask[:, m_idx] == 0] = np.mean(X_set[m_idx][Mask[:, m_idx] == 1], axis=0)
    
    return X_set, Y_onehot, Mask