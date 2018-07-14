from keras.regularizers import Regularizer
from keras import backend as K

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
        s_hat += 10 ** -5
        val = self.rho * K.log(self.rho/s_hat) + (1 - self.rho) * K.log((1 - self.rho)/(1 - s_hat))
        return self.beta*K.sum(val)

    def get_config(self):
        return {'beta': float(self.beta),
                'rho': float(self.rho)}

def kl_divergence(beta=1, rho=0.1):
    return KL_divergence(beta=beta, rho=rho)


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

    
