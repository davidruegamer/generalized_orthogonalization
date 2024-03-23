import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer

def orthog_tf(Y, X):
    Q = tf.linalg.qr(X, full_matrices=False, name="QR").q
    X_XtXinv_Xt = tf.linalg.matmul(Q, tf.linalg.matrix_transpose(Q))
    return(tf.subtract(Y, tf.tensordot(X_XtXinv_Xt, Y, [[1], [0]])))

class Orthogonalization(tf.keras.layers.Layer):
    def __init__(self, deactivate_at_test = True, **kwargs):
        self.deactivate_at_test = deactivate_at_test
        super(Orthogonalization, self).__init__(**kwargs)

    def call(self, Y, X, training=None):
        if not self.deactivate_at_test:
            return orthog_tf(Y, X)
        if training:
            return orthog_tf(Y, X)
        else:
            return Y
            
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'deactivate_at_test': self.deactivate_at_test
        })
        return config
        
class CustomOrthog(layers.Layer):
    def __init__(self):
        super(CustomOrthog, self).__init__()
        
    def call(self, inputs):
        yhat, train_red, ind = inputs
        
        yhat_shape = tf.shape(yhat)
        total_elements = tf.cast(tf.reduce_prod(yhat_shape), tf.int32)
        total_features = tf.cast(total_elements/yhat_shape[0], tf.int32)

        yhat = tf.reshape(yhat, [-1, 1])
        ind = tf.reshape(ind, [-1, 1])
        
        X_repeated = tf.tile(train_red, [total_features, 1]) 
        X_repeated = X_repeated * ind
        
        yhatc = orthog_tf(yhat, X_repeated)
        
        yhatc = tf.reshape(yhatc, yhat_shape)

        return yhatc
