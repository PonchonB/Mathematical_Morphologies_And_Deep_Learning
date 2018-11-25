from keras.constraints import Constraint
from keras import backend as K

class Between_0_and_1(Constraint):
    """Constrains the weights to be between 0 and 1.
    """
    def __call__(self, w):
        w = K.cast(K.clip(w, 0.,1), K.floatx())
        return w