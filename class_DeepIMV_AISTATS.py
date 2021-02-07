import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import fully_connected as FC_Net


_EPSILON = 1e-8

def div(x_, y_):
    return tf.div(x_, y_ + _EPSILON)

def log(x_):
    return tf.log(x_ + _EPSILON)

def xavier_initialization(size):
    dim_ = size[0]
    xavier_stddev = 1. / tf.sqrt(dim_ / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


### DEFINE PREDICTOR
def predictor(x_, o_dim_, o_type_, num_layers_=1, h_dim_=100, activation_fn=tf.nn.relu, keep_prob_=1.0, w_reg_=None):
    '''
        INPUT
            x_            : (2D-tensor) input
            o_dim_        : (int) output dimension
            o_type_       : (string) output type one of {'continuous', 'categorical', 'binary'}
            num_layers_   : (int) # of hidden layers
            activation_fn_: tf activation functions
        
        OUTPUT
            o_type_ tensor
    '''
    if o_type_ == 'continuous':
        out_fn = None
    elif o_type_ == 'categorical':
        out_fn = tf.nn.softmax #for classification task
    elif o_type_ == 'binary':
        out_fn = tf.nn.sigmoid
    else:
        raise ValueError('Wrong output type. The value {}!!'.format(o_type_))

    if num_layers_ == 1:
        out =  FC_Net(inputs=x_, num_outputs=o_dim_, activation_fn=out_fn, weights_regularizer=w_reg_, scope='out')
    else: #num_layers > 1
        for tmp_layer in range(num_layers_-1):
            if tmp_layer == 0:
                net = x_
            net = FC_Net(inputs=net, num_outputs=h_dim_, activation_fn=activation_fn, weights_regularizer=w_reg_, scope='layer_'+str(tmp_layer))
            net = tf.nn.dropout(net, keep_prob=keep_prob_)
        out =  FC_Net(inputs=net, num_outputs=o_dim_, activation_fn=out_fn, weights_regularizer=w_reg_, scope='out')  
    return out


### DEFINE STOCHASTIC ENCODER
def stochastic_encoder(x_, o_dim_, num_layers_=1, h_dim_=100, activation_fn=tf.nn.relu, keep_prob_=1.0, w_reg_=None):
    '''
        INPUT
            x_            : (2D-tensor) input
            o_dim_        : (int) output dimension
            num_layers_   : (int) # of hidden layers
            activation_fn_: tf activation functions
        
        OUTPUT
            [mu,sigma] tensor
    '''
    if num_layers_ == 1:
        out =  FC_Net(inputs=x_, num_outputs=o_dim_, activation_fn=None, weights_regularizer=w_reg_, scope='out')
    else: #num_layers > 1
        for tmp_layer in range(num_layers_-1):
            if tmp_layer == 0:
                net = x_
            net = FC_Net(inputs=net, num_outputs=h_dim_, activation_fn=activation_fn, weights_regularizer=w_reg_, scope='layer_'+str(tmp_layer))
            net = tf.nn.dropout(net, keep_prob=keep_prob_)
        out =  FC_Net(inputs=net, num_outputs=o_dim_, activation_fn=None, weights_regularizer=w_reg_, scope='out')  
    return out


### DEFINE SUPERVISED LOSS FUNCTION
def loss_y(y_true_, y_pred_, y_type_):                
    if y_type_ == 'continuous':
        tmp_loss = tf.reduce_sum((y_true_ - y_pred_)**2, axis=-1)
    elif y_type_ == 'categorical':
        tmp_loss = - tf.reduce_sum(y_true_ * log(y_pred_), axis=-1)
    elif y_type_ == 'binary':
        tmp_loss = - tf.reduce_sum(y_true_ * log(y_pred_) + (1.-y_true_) * log(1.-y_pred_), axis=-1)
    else:
        raise ValueError('Wrong output type. The value {}!!'.format(y_type_))                    
    return tmp_loss


### DEFINE NETWORK-RELATED FUNCTIONS
def product_of_experts(mask_, mu_set_, logvar_set_):
    tmp = 1.
    for m in range(len(mu_set_)):
        tmp += tf.reshape(mask_[:, m], [-1,1])*div(1., tf.exp(logvar_set_[m]))
    poe_var = div(1., tmp)
    poe_logvar = log(poe_var)
    
    tmp = 0.
    for m in range(len(mu_set_)):
        tmp += tf.reshape(mask_[:, m], [-1,1])*div(1., tf.exp(logvar_set_[m]))*mu_set_[m]
    poe_mu = poe_var * tmp
    
    return poe_mu, poe_logvar

    

###########################################################################
#### DEFINE PROPOSED-NETWORK
class DeepIMV_AISTATS:
    '''
        - Add mixture mode
        - Remove common/shared parts -- go back to the previous version
        - Leave the consistency loss; but make sure to set gamma = 0
    '''
    
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess             = sess
        self.name             = name
       
        # INPUT/OUTPUT DIMENSIONS
        self.M                = len(input_dims['x_dim_set'])
        
        self.x_dim_set = {}
        for m in range(self.M):
            self.x_dim_set[m] = input_dims['x_dim_set'][m]
            
        self.y_dim            = input_dims['y_dim']
        self.y_type           = input_dims['y_type']
       
        self.z_dim            = input_dims['z_dim']  # z_dim is equivalent to W and Z             
        self.steps_per_batch  = input_dims['steps_per_batch']
        
        # PREDICTOR INFO (VIEW-SPECIFC)
        self.h_dim_p1         = network_settings['h_dim_p1']      #predictor hidden nodes
        self.num_layers_p1    = network_settings['num_layers_p1'] #predictor layers
        
        # PREDICTOR INFO (MULTI_VIEW)
        self.h_dim_p2         = network_settings['h_dim_p2']      #predictor hidden nodes
        self.num_layers_p2    = network_settings['num_layers_p2'] #predictor layers
        
        # ENCODER INFO
        self.h_dim_e          = network_settings['h_dim_e']      #encoder hidden nodes
        self.num_layers_e     = network_settings['num_layers_e'] #encoder layers
       
        self.fc_activate_fn   = network_settings['fc_activate_fn'] 
        self.reg_scale        = network_settings['reg_scale']   #regularization
                
        self._build_net()
        
       
    def _build_net(self):
        ds     = tf.contrib.distributions
        
#         with tf.name_scope(self.name):
        with tf.variable_scope(self.name):
            self.mb_size        = tf.placeholder(tf.int32, [], name='batch_size')
            self.lr_rate        = tf.placeholder(tf.float32, name='learning_rate')           
            self.k_prob         = tf.placeholder(tf.float32, name='keep_probability')
                       
            ### INPUT/OUTPUT                   
            self.x_set          = {}
            for m in range(self.M):
                self.x_set[m]   = tf.placeholder(tf.float32, [None, self.x_dim_set[m]], 'input_{}'.format(m))
            
            self.mask           = tf.placeholder(tf.float32, [None, self.M], name='mask')            
            self.y              = tf.placeholder(tf.float32, [None, self.y_dim],  name='output')
                                   
            ### BALANCING COEFFICIENTS
            self.alpha          = tf.placeholder(tf.float32, name='coef_alpha') #Consitency Loss
            self.beta           = tf.placeholder(tf.float32, name='coef_beta')  #Information Bottleneck
            
            if self.reg_scale == 0:
                w_reg           = None
            else:
                w_reg           = tf.contrib.layers.l1_regularizer(scale=self.reg_scale)
                        
            ### PRIOR
            prior_z  = ds.Normal(0.0, 1.0) #PoE Prior - q(z)
            prior_z_set = {}
            for m in range(self.M):
                prior_z_set[m] = ds.Normal(0.0, 1.0) #View-Specific Prior - q(z_{m})
                        
            ### STOCHASTIC ENCODER
            self.h_set      = {}
            
            self.mu_z_set     = {}
            self.logvar_z_set = {}
            
            for m in range(self.M):
                with tf.variable_scope('encoder{}'.format(m+1)):
                    self.h_set[m]      = stochastic_encoder(
                        x_=self.x_set[m], o_dim_=2*self.z_dim, 
                        num_layers_=self.num_layers_e, h_dim_=self.h_dim_e, 
                        activation_fn=self.fc_activate_fn, keep_prob_=self.k_prob, w_reg_=w_reg
                    )
                    self.mu_z_set[m]     = self.h_set[m][:, :self.z_dim] 
                    self.logvar_z_set[m] = self.h_set[m][:, self.z_dim:]
            
            self.mu_z, self.logvar_z = product_of_experts(self.mask, self.mu_z_set, self.logvar_z_set)                
            
            qz         = ds.Normal(self.mu_z, tf.sqrt(tf.exp(self.logvar_z)))
            self.z     = qz.sample()
            self.zs    = qz.sample(10)

            qz_set     = {}
            self.z_set = {}
            for m in range(self.M):
                qz_set[m]      = ds.Normal(self.mu_z_set[m], tf.sqrt(tf.exp(self.logvar_z_set[m])))
                self.z_set[m]  = qz_set[m].sample()

    
    
            ### PREDICTOR (JOINT)
            with tf.variable_scope('predictor'):
                self.y_hat = predictor(
                    x_=self.z, o_dim_=self.y_dim, o_type_=self.y_type, 
                    num_layers_=self.num_layers_p2, h_dim_=self.h_dim_p2, 
                    activation_fn=self.fc_activate_fn, keep_prob_=self.k_prob, w_reg_=w_reg
                )

            # this will generate multiple samples of y (based on multiple samples drawn from the variational encoder.
            with tf.variable_scope('predictor', reuse=True):               
                self.y_hats = predictor(
                    x_=self.zs, o_dim_=self.y_dim, o_type_=self.y_type, 
                    num_layers_=self.num_layers_p2, h_dim_=self.h_dim_p2, 
                    activation_fn=self.fc_activate_fn, keep_prob_=self.k_prob, w_reg_=w_reg
                )
                                    
            ### PREDICTOR 
            self.y_hat_set = {}
            for m in range(self.M):
                with tf.variable_scope('predictor_set{}'.format(m)):
                    self.y_hat_set[m] = predictor(
                        x_=self.z_set[m], o_dim_=self.y_dim, o_type_=self.y_type, 
                        num_layers_=self.num_layers_p1, h_dim_=self.h_dim_p1, 
                        activation_fn=self.fc_activate_fn, keep_prob_=self.k_prob, w_reg_=w_reg
                    )

            
            ### OPTIMIZER
            global_vars      = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            enc_vars         = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/encoder')
            pred_vars        = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/predictor')
            
        
            ### CONSITENCY LOSS
            self.LOSS_CONSISTENCY = 0.
            for m in range(self.M):
                self.LOSS_CONSISTENCY += 1./self.M * div(
                    tf.reduce_sum(self.mask[:, m] * tf.reduce_sum(ds.kl_divergence(qz, qz_set[m]), axis=-1)),
                    tf.reduce_sum(self.mask[:, m])
                )
                                                                        

            self.LOSS_KL       = tf.reduce_mean(
                tf.reduce_sum(ds.kl_divergence(qz, prior_z), axis=-1)
            )            
            self.LOSS_P        = tf.reduce_mean(loss_y(self.y, self.y_hat, self.y_type))
            
            self.LOSS_IB_JOINT = self.LOSS_P + self.beta*self.LOSS_KL
                                    
            self.LOSS_Ps_all  = []
            self.LOSS_KLs_all = []
            for m in range(self.M):
                tmp_p              = loss_y(self.y, self.y_hat_set[m], self.y_type)
                tmp_kl             = tf.reduce_sum(ds.kl_divergence(qz_set[m], prior_z_set[m]), axis=-1)
        
                self.LOSS_Ps_all  += [div(tf.reduce_sum(self.mask[:,m]*tmp_p), tf.reduce_sum(self.mask[:,m]))]        
                self.LOSS_KLs_all += [div(tf.reduce_sum(self.mask[:,m]*tmp_kl), tf.reduce_sum(self.mask[:,m]))]
        
                
            self.LOSS_Ps_all  = tf.stack(self.LOSS_Ps_all, axis=0)
            self.LOSS_KLs_all = tf.stack(self.LOSS_KLs_all, axis=0)
            
            self.LOSS_Ps      = tf.reduce_sum(self.LOSS_Ps_all)
            self.LOSS_KLs     = tf.reduce_sum(self.LOSS_KLs_all)
                        
            self.LOSS_IB_MARGINAL = self.LOSS_Ps + self.beta*self.LOSS_KLs         
            

            self.LOSS_TOTAL       = self.LOSS_IB_JOINT\
                                    + self.alpha*(self.LOSS_IB_MARGINAL)\
                                    + tf.losses.get_regularization_loss()
    
            
            self.global_step      = tf.contrib.framework.get_or_create_global_step()
            self.lr_rate_decayed  = tf.train.exponential_decay(self.lr_rate, self.global_step,
                                                       decay_steps=2*self.steps_per_batch,
                                                       decay_rate=0.97, staircase=True)

            opt                = tf.train.AdamOptimizer(self.lr_rate_decayed, 0.5)
                
                        
            ma = tf.train.ExponentialMovingAverage(0.999, zero_debias=True)
            ma_update = ma.apply(tf.model_variables())
            

            self.solver = tf.contrib.training.create_train_op(self.LOSS_TOTAL, opt,
                                                               self.global_step,
                                                               update_ops=[ma_update])
                

    def train(self, x_set_, y_, m_, alpha_, beta_, lr_train, k_prob=1.0):
        feed_dict_ = self.make_feed_dict(x_set_)
        feed_dict_.update({self.y: y_, self.mask: m_, 
                           self.alpha: alpha_, self.beta: beta_, 
                           self.mb_size: np.shape(x_set_[0])[0],
                           self.lr_rate: lr_train, self.k_prob: k_prob})        
        return self.sess.run([self.solver, self.LOSS_TOTAL, self.LOSS_P, self.LOSS_KL, self.LOSS_Ps, 
                              self.LOSS_KLs, self.LOSS_CONSISTENCY],
                             feed_dict=feed_dict_)
   
    def get_loss(self, x_set_, y_, m_, alpha_, beta_):
        feed_dict_ = self.make_feed_dict(x_set_)
        feed_dict_.update({self.y: y_, self.mask: m_, 
                           self.alpha: alpha_, self.beta: beta_,
                           self.mb_size: np.shape(x_set_[0])[0], self.k_prob: 1.0})        
        return self.sess.run([self.LOSS_TOTAL, self.LOSS_P, self.LOSS_KL, self.LOSS_Ps, 
                              self.LOSS_KLs, self.LOSS_CONSISTENCY, self.LOSS_Ps_all, self.LOSS_KLs_all],
                             feed_dict=feed_dict_)
        

    def predict_y(self, x_set_, m_):
        feed_dict_ = self.make_feed_dict(x_set_)
        feed_dict_.update({self.mask: m_, self.mb_size: np.shape(x_set_[0])[0], self.k_prob: 1.0})
        return self.sess.run(self.y_hat, feed_dict=feed_dict_)
    
    def predict_ys(self, x_set_, m_):
        feed_dict_ = self.make_feed_dict(x_set_)
        feed_dict_.update({self.mask: m_, self.mb_size: np.shape(x_set_[0])[0], self.k_prob: 1.0})
        return self.sess.run([self.y_hat, self.y_hats], feed_dict=feed_dict_)

    def predict_yhat_set(self, x_set_, m_):
        feed_dict_ = self.make_feed_dict(x_set_)
        feed_dict_.update({self.mask: m_, self.mb_size: np.shape(x_set_[0])[0], self.k_prob: 1.0})
        return self.sess.run(self.y_hat_set, feed_dict=feed_dict_)
    
    def predict_mu_z_and_mu_z_set(self, x_set_, m_): #this outputs mu and mu_set
        feed_dict_ = self.make_feed_dict(x_set_)
        feed_dict_.update({self.mask: m_, self.mb_size: np.shape(x_set_[0])[0], self.k_prob: 1.0})
        return self.sess.run([self.mu_z, self.mu_z_set], feed_dict=feed_dict_)
    
    def predict_logvar_z_and_logvar_z_set(self, x_set_, m_): #this outputs sigma and sigma_set
        feed_dict_ = self.make_feed_dict(x_set_)
        feed_dict_.update({self.mask: m_, self.mb_size: np.shape(x_set_[0])[0], self.k_prob: 1.0})
        return self.sess.run([self.logvar_z, self.logvar_z_set], feed_dict=feed_dict_)

    def predict_z_n_z_set(self, x_set_, m_):  #this outputs z and z_set
        feed_dict_ = self.make_feed_dict(x_set_)
        feed_dict_.update({self.mask: m_, self.mb_size: np.shape(x_set_[0])[0], self.k_prob: 1.0})
        return self.sess.run([self.z, self.z_set], feed_dict=feed_dict_)
  
    def make_feed_dict(self, x_set_):
        feed_dict_ = {}
        for m in range(len(self.x_set)):
            feed_dict_[self.x_set[m]] = x_set_[m]           
        return feed_dict_

    