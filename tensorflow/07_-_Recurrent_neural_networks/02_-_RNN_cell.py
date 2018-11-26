import tensorflow as tf
import numpy as np

class Cell(tf.nn.rnn_cell.RNNCell):
    def __init__(self):
        super(Cell, self).__init__()

    @property
    def state_size(self):
        return 2

    @property
    def output_size(self):
        return 2

    def build(self, inputs_shape):
        print('inputs_shape:', inputs_shape)
        self.built = True
        
    def call(self, x, curr_state):
        print('curr_state:', curr_state)
        print('x:', x)
        new_state = x
        output = -x
        return (output, new_state)


g = tf.Graph()
with g.as_default():
    init_state = tf.constant([[0, 0]], tf.float32, [1, 2], 'init')

    c = Cell()
    (output_cell, state_cell) = c.call(tf.constant([1, 1], tf.float32, [2]), init_state)

    seqs = tf.constant(
        [
            [[1, 1], [2, 2], [3, 3]],
        ],
        tf.float32,
        [1, 3, 2],
        'seqs'
    )
    cell = Cell()
    (outputs, state) = tf.nn.dynamic_rnn(cell, seqs, initial_state=init_state)

    g.finalize()

    with tf.Session() as s:
        [ curr_output_cell, curr_state_cell, curr_outputs, curr_state ] = s.run([ output_cell, state_cell, outputs, state ], { })
        print()
        print('output_cell:')
        print(curr_output_cell)
        print()
        print('state_cell:')
        print(curr_state_cell)
        print()
        print('outputs:')
        print(curr_outputs)
        print()
        print('state:')
        print(curr_state)
