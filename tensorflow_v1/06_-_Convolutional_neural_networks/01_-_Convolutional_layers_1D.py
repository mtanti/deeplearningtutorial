import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

###################################

class Model(object):

    def __init__(self):
        kernel_width = 3 #The size of the kernel sliding over the inputs.
        stride = 1 #The step size to move the kernel as it's sliding.
        in_vec_size = 5 #The vector sizes in the inputs.
        out_vec_size = 2 #The transformed vector sizes in the outputs.

        self.graph = tf.Graph()
        with self.graph.as_default():
            #A single input consisting of vectors of size 5 arranged in a sequence of length 4.
            seqs = tf.constant(
                    [
                        [1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]
                    ], tf.float32, [1, 4, in_vec_size], 'seqs'
                )
    
            #The kernel.
            kernel = tf.constant(
                    [
                        [ 1,  2], [ 3,  4], [ 5,  6], [ 7,  8], [ 9, 10],
                        [11, 12], [13, 14], [15, 16], [17, 18], [19, 20],
                        [21, 22], [23, 24], [25, 26], [27, 28], [29, 30]
                    ], tf.float32, [kernel_width, in_vec_size, out_vec_size], 'kernel'
                )
            
            #The convolution
            self.conved_seqs = tf.nn.conv1d(seqs, kernel, stride, 'VALID')

            self.graph.finalize()

            self.sess = tf.Session()

    def close(self):
        self.sess.close()
    
    def convolve(self):
        return self.sess.run([ self.conved_seqs ], {  })[0]
    
###################################
model = Model()
print(model.convolve())