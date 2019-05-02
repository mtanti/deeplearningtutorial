import numpy as np
import tensorflow as tf

window_width = 2 #The size of the window sliding over the inputs
stride = 1 #The step size to move the window as it's sliding
in_vec_size = 2 #The vector sizes in the inputs
out_vec_size = 3 #The transformed vector sizes in the outputs
padding = 'VALID' #Whether to automatically pad the input sequence or not to avoid getting a smaller output sequence ('SAME':pad, 'VALID':no pad)

g = tf.Graph()
with g.as_default():
    #A single input consisting of vectors of size 2 arranged in a sequence of length 5
    seqs = tf.constant(
        [
            [[ 1, 2], [ 3, 4], [ 5, 6], [ 7, 8], [ 9,10]],
        ], tf.float32, [1, 5, in_vec_size], 'seqs'
    )
    
    #The weights with which to multiply each number in each vector in a captured window before summing the weighted numbers
    kernel = tf.constant(
        [ #Each row handles a different part of the window of width 2
            [[ 1, 2, 3],[ 4, 5, 6]],
            [[ 7, 8, 9],[10,11,12]],
        ], tf.float32, [window_width, in_vec_size, out_vec_size], 'kernel'
    )
    
    #The convolution
    conved_seqs = tf.nn.conv1d(seqs, kernel, stride, padding)

    g.finalize()

    with tf.Session() as s:
        [ new_seq ] = s.run([ conved_seqs ], { })
        print(new_seq)
