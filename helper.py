import numpy as np
import random

from sklearn.metrics import roc_auc_score, average_precision_score

def f_get_minibatch_set(mb_size_, x_set_, y_, m_):
    idx = range(np.shape(y_)[0])
    idx = random.sample(idx, mb_size_)
    
    x_set_mb = {}
    for m in range(len(x_set_)):
        x_set_mb[m] = x_set_[m][idx].astype(float)    
    y_mb   = y_[idx].astype(float)
    m_mb  = m_[idx].astype(float)

    return x_set_mb, y_mb, m_mb


def evaluate(true_y_, pred_y_, y_type_):
    if y_type_ == 'categorical':
        acc = np.mean(np.argmax(true_y_, axis=1) == np.argmax(pred_y_, axis=1))
        return acc
    elif y_type_ == 'binary':
        auroc = roc_auc_score(true_y_[:, 1], pred_y_[:,1])
        auprc = average_precision_score(true_y_[:, 1], pred_y_[:,1])
        return auroc, auprc
    elif y_type_ == 'continuous':
        mse   = np.mean((true_y_.reshape([-1]) - pred_y_.reshape([-1]))**2)
        return mse
    else:
        ValueError(print("y_type should be {'categorical', 'binary', 'continuous'}"))