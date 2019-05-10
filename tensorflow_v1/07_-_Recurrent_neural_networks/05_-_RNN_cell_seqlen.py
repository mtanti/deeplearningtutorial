import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


class Cell(tf.nn.rnn_cell.RNNCell):
    def __init__(self):
        super().__init__()

    @property
    def state_size(self):
        return 2

    @property
    def output_size(self):
        return 2

    def build(self, inputs_shape):
        self.built = True

    def call(self, x, curr_state):
        new_state = x
        output = -x
        return (output, new_state)


g = tf.Graph()
with g.as_default():
    init_state = tf.constant(
        [
            [ 0, 0 ]
        ], tf.float32, [1, 2], 'init'
    )

    seqs = tf.constant(
        [
            [ [1, 1], [2, 2], [3, 3] ],
        ], tf.float32, [1, 3, 2], 'seqs'
    )
    seq_len = tf.constant(
        [
            2, #The length of the sequence is two input vectors long, with any other vectors after that to be ignored
        ], tf.float32, [1]
    )
    cell = Cell()
    (outputs, state) = tf.nn.dynamic_rnn(cell, seqs, sequence_length=seq_len, initial_state=init_state)

    g.finalize()

    with tf.Session() as s:
        [ curr_outputs, curr_state ] = s.run([ outputs, state ], { })
        
        print('outputs:')
        print(curr_outputs)
        print()
        
        print('state:')
        print(curr_state)
