import tensorflow as tf
import numpy as np

embedding_size = 2
state_size = 2

class Cell(tf.nn.rnn_cell.RNNCell):
    def __init__(self):
        super(Cell, self).__init__()
        self.W = None
        self.b = None

    @property
    def state_size(self):
        return state_size

    @property
    def output_size(self):
        return (state_size + embedding_size, state_size)

    def build(self, inputs_shape):
        self.W = self.add_variable('W', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=0.1, seed=0))
        self.b = self.add_variable('b', [state_size], tf.float32, tf.zeros_initializer())
        self.built = True
        
    def call(self, x, curr_state):
        state_input = tf.concat([ curr_state, x ], axis=1)
        new_state = tf.tanh(tf.matmul(state_input, self.W) + self.b)
        return ((state_input, new_state), new_state)

g = tf.Graph()
with g.as_default():
    init_state = tf.constant([[0, 0]], tf.float32, [1, 2], 'init')

    seqs = tf.constant(
        [
            [[1, 1], [2, 2], [3, 3]],
        ],
        tf.float32,
        [1, 3, 2],
        'seqs'
    )
    
    seq_len = tf.constant(
        [
            2,
        ],
        tf.float32,
        [1]
    )
    
    cell = Cell()
    ((state_input, outputs), state) = tf.nn.dynamic_rnn(cell, seqs, sequence_length=seq_len, initial_state=init_state)
    
    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })
        
        [
            curr_state_input, curr_outputs, curr_state
        ] = s.run([
            state_input, outputs, state
        ], { })

        print('-------------')
        print('curr_state_input')
        print(curr_state_input)
        print()
        
        print('-------------')
        print('curr_outputs')
        print(curr_outputs)
        print()
        
        print('-------------')
        print('curr_state')
        print(curr_state)
        print()
