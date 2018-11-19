import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    seqs = tf.constant(
        np.array([
            [[ 1, 2], [ 3, 4], [ 5, 6], [ 7, 8]],
        ]),
        tf.float32,
        [1, 4, 2] #A single input consisting of vectors of size 2 arranged in a sequence of length 4
    )
    
    window_width = 2 #The size of the window sliding over the inputs
    stride = 1 #The amount to move the window as it's sliding
    in_vec_size = 2 #The vector sizes in the inputs
    out_vec_size = 1 #The transformed vector sizes in the outputs
    padding = 'VALID' #Whether to automatically pad the input sequence or not to avoid getting a smaller output sequence ('SAME':pad, 'VALID':no pad)
    kernel = tf.constant(
        np.array([
            [[0],[0]], [[1],[0]],
        ]),
        tf.float32,
        [window_width, in_vec_size, out_vec_size]
    ) #The weights with which to multiply each number in each vector in a captured window before summing the weighted numbers
    
    conved_seqs = tf.nn.conv1d(seqs, kernel, stride, padding)

    g.finalize()

    with tf.Session() as s:
        [new_seq] = s.run(conved_seqs)
        print(new_seq)
