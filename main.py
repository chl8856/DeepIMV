import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

import random
import sys, os

from sklearn.model_selection import train_test_split

import import_data as impt
from helper import f_get_minibatch_set, evaluate
from class_DeepIMV_AISTATS import DeepIMV_AISTATS


import argparse

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1234, help='random seed', type=int)
    
    parser.add_argument('--h_dim_p', default=100, help='number of hidden nodes -- predictor', type=int)
    parser.add_argument('--num_layers_p', default=2, help='number of layers -- predictor', type=int)

    parser.add_argument('--h_dim_e', default=100, help='number of hidden nodes -- encoder', type=int)
    parser.add_argument('--num_layers_e', default=3, help='number of layers -- encoder', type=int)
    
    parser.add_argument('--z_dim', default=50, help='dimension of latent representations', type=int)

    
    parser.add_argument("--lr_rate", default=1e-4, help='learning rate', type=float)
    parser.add_argument("--l1_reg", default=0., help='l1-regularization', type=float)

    parser.add_argument("--itrs", default=50000, type=int)
    parser.add_argument("--step_size", default=1000, type=int)
    parser.add_argument("--max_flag", default=20, type=int)

    parser.add_argument("--mb_size", default=32, type=int)
    parser.add_argument("--keep_prob", help='keep probability for dropout', default=0.7, type=float)
    
    parser.add_argument('--alpha', default=1.0, help='coefficient -- alpha', type=float)
    parser.add_argument('--beta', default=0.01, help='coefficient -- beta', type=float)
    
    parser.add_argument('--save_path', default='./storage/', help='path to save files', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    
    args             = init_arg()    
    seed             = args.seed
    ### import multi-view dataset with arbitrary view-missing patterns.
    X_set, Y_onehot, Mask = impt.import_incomplete_handwritten()
    
    tr_X_set, te_X_set, va_X_set = {}, {}, {}

    # 64/16/20 training/validation/testing split
    for m in range(len(X_set)):
        tr_X_set[m],te_X_set[m] = train_test_split(X_set[m], test_size=0.2, random_state=seed)   
        tr_X_set[m],va_X_set[m] = train_test_split(tr_X_set[m], test_size=0.2, random_state=seed)

    tr_Y_onehot,te_Y_onehot, tr_M,te_M = train_test_split(Y_onehot, Mask, test_size=0.2, random_state=seed)
    tr_Y_onehot,va_Y_onehot, tr_M,va_M = train_test_split(tr_Y_onehot, tr_M, test_size=0.2, random_state=seed)

    x_dim_set    = [tr_X_set[m].shape[1] for m in range(len(tr_X_set))]
    y_dim        = np.shape(tr_Y_onehot)[1]

    if y_dim == 1:
        y_type       = 'continuous'
    elif y_dim == 2:
        y_type       = 'binary'
    else:
        y_type       = 'categorical'
    
    
    mb_size         = args.mb_size
    steps_per_batch = int(np.shape(tr_M)[0]/mb_size) #for moving average
    
    input_dims = {
        'x_dim_set': x_dim_set,
        'y_dim': y_dim,
        'y_type': y_type,
        'z_dim': args.z_dim,

        'steps_per_batch': steps_per_batch
    }

    network_settings = {
        'h_dim_p1': args.h_dim_p,
        'num_layers_p1': args.num_layers_p,   #view-specific

        'h_dim_p2': args.h_dim_p,
        'num_layers_p2': args.num_layers_p,  #multi-view

        'h_dim_e': args.h_dim_e,
        'num_layers_e': args.num_layers_e,

        'fc_activate_fn': tf.nn.relu,
        'reg_scale': args.l1_reg,
    }
    

    lr_rate         = args.lr_rate
    iteration       = args.itrs
    stepsize        = args.step_size
    max_flag        = args.max_flag

    k_prob          = args.keep_prob
    
    alpha           = args.alpha
    beta            = args.beta
    
    save_path       = args.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    tf.reset_default_graph()
    gpu_options = tf.GPUOptions()
    
    sess  = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    model = DeepIMV_AISTATS(sess, "DeepIMV_AISTATS", input_dims, network_settings)
    

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    ##### TRAINING
    min_loss  = 1e+8   
    max_acc   = 0.0

    tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc = 0, 0, 0, 0, 0, 0
    va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc = 0, 0, 0, 0, 0, 0
    
    stop_flag = 0
    for itr in range(iteration):
        x_mb_set, y_mb, m_mb          = f_get_minibatch_set(mb_size, tr_X_set, tr_Y_onehot, tr_M)     

        _, Lt, Lp, Lkl, Lps, Lkls, Lc = model.train(x_mb_set, y_mb, m_mb, alpha, beta, lr_rate, k_prob)

        tr_avg_Lt   += Lt/stepsize
        tr_avg_Lp   += Lp/stepsize
        tr_avg_Lkl  += Lkl/stepsize
        tr_avg_Lps  += Lps/stepsize
        tr_avg_Lkls += Lkls/stepsize
        tr_avg_Lc   += Lc/stepsize


        x_mb_set, y_mb, m_mb          = f_get_minibatch_set(min(np.shape(va_M)[0], mb_size), va_X_set, va_Y_onehot, va_M)       
        Lt, Lp, Lkl, Lps, Lkls, Lc, _, _    = model.get_loss(x_mb_set, y_mb, m_mb, alpha, beta)

        va_avg_Lt   += Lt/stepsize
        va_avg_Lp   += Lp/stepsize
        va_avg_Lkl  += Lkl/stepsize
        va_avg_Lps  += Lps/stepsize
        va_avg_Lkls += Lkls/stepsize
        va_avg_Lc   += Lc/stepsize

        if (itr+1)%stepsize == 0:
            y_pred, y_preds = model.predict_ys(va_X_set, va_M)

    #         score = 

            print( "{:05d}: TRAIN| Lt={:.3f} Lp={:.3f} Lkl={:.3f} Lps={:.3f} Lkls={:.3f} Lc={:.3f} | VALID| Lt={:.3f} Lp={:.3f} Lkl={:.3f} Lps={:.3f} Lkls={:.3f} Lc={:.3f} score={}".format(
                itr+1, tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc,  
                va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc, evaluate(va_Y_onehot, np.mean(y_preds, axis=0), y_type))
                 )

            if min_loss > va_avg_Lt:
                min_loss  = va_avg_Lt
                stop_flag = 0
                saver.save(sess, save_path  + 'best_model')
                print('saved...')
            else:
                stop_flag += 1

            tr_avg_Lt, tr_avg_Lp, tr_avg_Lkl, tr_avg_Lps, tr_avg_Lkls, tr_avg_Lc = 0, 0, 0, 0, 0, 0
            va_avg_Lt, va_avg_Lp, va_avg_Lkl, va_avg_Lps, va_avg_Lkls, va_avg_Lc = 0, 0, 0, 0, 0, 0

            if stop_flag >= max_flag:
                break

    print('FINISHED...')
    
    
    ##### TESTING
    saver.restore(sess, save_path  + 'best_model')
    
    _, pred_ys = model.predict_ys(te_X_set, te_M)
    pred_y = np.mean(pred_ys, axis=0)

    print('Test Score: {}'.format(evaluate(te_Y_onehot, pred_y, y_type)))