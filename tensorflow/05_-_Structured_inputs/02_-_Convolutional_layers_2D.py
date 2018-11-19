import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    grids = tf.constant(
        np.array([
            [
                [[ 1, 2], [ 3, 4], [ 5, 6]],
                [[ 7, 8], [ 9,10], [11,12]],
                [[13,14], [15,16], [17,18]],
            ]
        ]),
        tf.float32,
        [1, 3, 3, 2] #A single input consisting of a vectors of size 2 arranged in a 3x3 grid
    )
    
    window_width = 2 #The width of the window sliding over the inputs
    window_height = 2 #The height of the window sliding over the inputs
    stride_x = 1 #The amount to move the window horizontally as it's sliding
    stride_y = 1 #The amount to move the window vertically as it's sliding
    in_vec_size = 2
    out_vec_size = 1
    padding = 'VALID'
    kernel = tf.constant(
        np.array([
            [[[0],[0]], [[0],[0]]],
            [[[1],[0]], [[0],[0]]],
        ]),
        tf.float32,
        [window_width, window_height, in_vec_size, out_vec_size]
    )
    
    conved_grids = tf.nn.conv2d(grids, kernel, [1, stride_x, stride_y, 1], padding) #Note that the 1s in the stride list are for skipping grids in the batch and for skipping elements in vectors, which are things that we generally don't want so we leave them as 1 to not skip anything

    g.finalize()

    with tf.Session() as s:
        [new_grid] = s.run(conved_grids)
        print(new_grid)
