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

    #This is where we specify the variables being used by the cell (if we had any)
    def build(self, input_shape):
        print('build - input_shape:', input_shape[0].value, input_shape[1].value)
        self.built = True

    #This is where you say how to combine the input and state vectors (note: the state, not the output vectors)
    #This function defines the new state as s(t+1) = x(t) and the new output as o(t+1) = -x(t)
    def call(self, x, curr_state):
        #After running this script, notice the following in the print:
        # * The dynamic_rnn changes the names of the x and curr_state
        # * The shape of x would not include the sequence size as we only work with a single input item at a time
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
    (output_cell, state_cell) = c(input_vec, init_state) #Just treat the cell as if it is a function

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