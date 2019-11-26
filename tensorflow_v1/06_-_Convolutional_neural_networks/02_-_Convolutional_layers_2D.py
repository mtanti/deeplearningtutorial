import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

###################################

class Model(object):

    def __init__(self):
        kernel_width = 2 #The width of the kernel sliding over the inputs.
        kernel_height = 4 #The height of the kernel sliding over the inputs.
        stride_x = 1 #The step size to move the kernel in the x direction as it's sliding.
        stride_y = 1 #The step size to move the kernel in the y direction as it's sliding.
        in_vec_size = 3 #The vector sizes in the inputs.
        out_vec_size = 1 #The transformed vector sizes in the outputs.

        self.graph = tf.Graph()
        with self.graph.as_default():
            #A single input consisting of vectors of size 3 arranged in a grid of width 4 and height 5.
            grids = tf.constant(
                    [
                        [[ 1,  2,  3], [ 4,  5,  6], [ 7,  8,  9], [10, 11, 12]],
                        [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
                        [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 35, 36]],
                        [[37, 38, 39], [40, 41, 42], [43, 44, 45], [46, 47, 48]],
                        [[49, 50, 51], [52, 53, 54], [55, 56, 57], [58, 59, 60]]
                    ], tf.float32, [1, 5, 4, in_vec_size], 'grids'
                )
    
            #The kernel.
            kernel = tf.constant(
                    [
                        [[[ 1], [ 2], [ 3]],  [[ 4], [ 5], [ 6]]],
                        [[[ 7], [ 8], [ 9]],  [[10], [11], [12]]],
                        [[[13], [14], [15]],  [[16], [17], [18]]],
                        [[[19], [20], [21]],  [[22], [23], [24]]]
                    ], tf.float32, [kernel_height, kernel_width, in_vec_size, out_vec_size], 'kernel'
                )
            
            #The convolution
            #Note that the 1s in the stride list are for skipping grids in the batch and for skipping elements in vectors, which are things that we generally don't want so we leave them as 1 to not skip anything.
            self.conved_grids = tf.nn.conv2d(grids, kernel, [1,stride_y,stride_x,1], 'VALID')

            self.graph.finalize()

            self.sess = tf.Session()

    def close(self):
        self.sess.close()
    
    def convolve(self):
        return self.sess.run([ self.conved_grids ], {  })[0]
    
###################################
model = Model()
print(model.convolve())