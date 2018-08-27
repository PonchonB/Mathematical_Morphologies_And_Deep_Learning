from keras.regularizers import Regularizer
from keras import backend as K
import math

class KL_divergence(Regularizer):
    """KL divergence for Sparsity regularization.
    # Arguments
        beta: Float; Weight of the kl_regularizer.
        rho: Float; Sparsity Parameter.
    """

    def __init__(self, beta=1, rho=0.1):
        self.beta = K.cast_to_floatx(beta)
        self.rho = K.cast_to_floatx(rho)

    def __call__(self, x):
        s_hat = K.mean(x, 0)
        s_hat += 10 ** -5
        val = self.rho * K.log(self.rho/s_hat) + (1 - self.rho) * K.log((1 - self.rho)/(1 - s_hat))
        return self.beta*val

    def get_config(self):
        return {'beta': float(self.beta),
                'rho': float(self.rho)}

    
class KL_divergence_sum(Regularizer):
    """KL divergence for Sparsity regularization.
    # Arguments
        beta: Float; Weight of the kl_regularizer.
        rho: Float; Sparsity Parameter.
    """

    def __init__(self, beta=1, rho=0.1):
        self.beta = K.cast_to_floatx(beta)
        self.rho = K.cast_to_floatx(rho)

    def __call__(self, x):
        s_hat = K.mean(x, 0)
        s_hat = K.clip(s_hat, K.epsilon(), 1)
        val = self.rho * K.log(self.rho/s_hat) + (1 - self.rho) * K.log((1 - self.rho)/(1 - s_hat))
        return self.beta*K.sum(val)

    def get_config(self):
        return {'beta': float(self.beta),
                'rho': float(self.rho)}

def kl_divergence(beta=1, rho=0.1):
    return KL_divergence(beta=beta, rho=rho)

class L1(Regularizer):
    """L1 norm for Sparsity regularization. 
    Takes the mean of the L1 norm of the activity on the batch, instead of the sum as done in Keras (batch size dependant)
    # Arguments
        beta: Float; Weight of the L1_regularizer.
    """

    def __init__(self, beta=1):
        self.beta = K.cast_to_floatx(beta)

    def __call__(self, x):
        val = K.mean(K.sum(K.abs(x), axis=1))
        return self.beta*val

    def get_config(self):
        return {'beta': float(self.beta)}
        
class sparseness(Regularizer):
    """
    Using the sparseness measure from Hoyer 2004 (Sparse-NMF) to enforce sparsity.
    # Arguments
        beta: Float; Weight of the regularizer in the total cost function to be minimized.
        sp_o: Float in [0,1], sparsity objective of the measure (the higher the more sparse).
    """
    def __init__(self, beta=1, sp_o=0.6):
        self.beta = K.cast_to_floatx(beta)
        self.sp_o = K.cast_to_floatx(sp_o)

    def __call__(self, x):
        dim = K.int_shape(x)[1]
        l1_l2_ratio = K.sum(K.abs(x), axis=1)/(K.sqrt(K.sum(K.square(x), axis=1)) + K.epsilon())
        sqrt = math.sqrt(dim) 
        sp = (sqrt - l1_l2_ratio)/(sqrt - 1 + K.epsilon())
        return self.beta*K.mean(K.abs(self.sp_o - sp))
    
    def get_config(self):
        return {'beta': float(self.beta), 'sp_o':float(self.sp_o)}


class asymmetric_weight_decay(Regularizer):
    """Asymmetric weight decay for Non_Negativity constraint.
    # Arguments
        alpha: Float; Weight decay parameter for the positive weights.
        beta: Float; Weight decay parameter for the negative weights.
        lam: Float; Weight of the whole regularizer loss.
    """

    def __init__(self, alpha=0.1, beta=1, lam=0.1):
        self.alpha = K.cast_to_floatx(alpha)
        self.beta = K.cast_to_floatx(beta)
        self.lam = K.cast_to_floatx(lam)

    def __call__(self, x):
        sh = K.shape(x)
        neg = K.cast(K.less(x, K.zeros(sh)), 'float32')
        pos = K.cast(K.less(K.zeros(sh), x), 'float32')
        sq = K.square(x)
        val = K.sum(self.beta*neg*sq + self.alpha*pos*sq)
        return self.lam/2*val

    def get_config(self):
        return {'alpha': float(self.alpha),
                'beta': float(self.beta),
                'lam': float(self.lam)}

    
