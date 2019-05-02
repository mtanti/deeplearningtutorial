import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


#The cell object that defines how the input and state combine into the next state
class Cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self):
        super().__init__()

    #The vector size of the state
    @property
    def state_size(self):
        return 2

    #The vector size of the final output of the RNN
    @property
    def output_size(self):
        return 2

    #This is where you define any variables used by the RNN cell
    def build(self, inputs_shape):
        print('building')
        print('inputs_shape:', inputs_shape)
        print()

    #This is where you say how to combine the input and state vectors (note: the state, not the output vectors)
    #This function defines the new state as s(t+1) = x(t) and the new output as o(t+1) = -x(t)
    def call(self, x, curr_state):
        print('curr_state:', curr_state)
        print('x:', x)
        print()
        
        new_state = x
        new_output = -x
        return (new_output, new_state)


g = tf.Graph()
with g.as_default():
    #The initial state must be defined as a batch of initial states for each item in the minibatch
    #Use tf.tile to replicate the same initial state vector for each item in the batch
    init_state = tf.constant(
        [
            [ 0, 0 ],
        ], tf.float32, [1, 2], 'init'
    )

    #Using the RNN cell for a single input-state combination
    input_vec = tf.constant(
        [
            1,
            1,
        ],
        tf.float32, [2]
    )
    c = Cell()
    (output_cell, state_cell) = c.call(input_vec, init_state)

    #Using the RNN cell to process a whole sequence of inputs
    seqs = tf.constant(
        [
            [ [1, 1], [2, 2], [3, 3] ],
        ], tf.float32, [1, 3, 2], 'seqs'
    )
    cell = Cell()
    (outputs, state) = tf.nn.dynamic_rnn(cell, seqs, initial_state=init_state)

    g.finalize()

    with tf.Session() as s:
        [ curr_output_cell, curr_state_cell, curr_outputs, curr_state ] = s.run([ output_cell, state_cell, outputs, state ], { })

        print('-------')
        print('output_cell:')
        print(curr_output_cell)
        print()

        print('state_cell:')
        print(curr_state_cell)
        print()

        print('-------')
        print('outputs:')
        print(curr_outputs)
        print()

        print('state:')
        print(curr_state)