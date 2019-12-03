import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np

###################################
class Cell(tf.nn.rnn_cell.RNNCell):
    #The cell object that defines how the input and state combine into the next state.

    def __init__(self):
        super().__init__()

    #The vector size of the RNN state.
    @property
    def state_size(self):
        return 2

    #The vector size of the RNN output.
    @property
    def output_size(self):
        return 2

    #This is where we specify the variables being used by the cell (if we had any).
    def build(self, input_shape):
        print('build - input_shape:', input_shape)
        self.built = True

    #This is where you say how to combine the input and state vectors (note: the state, not the output vectors).
    #This function defines the new state as s(t+1) = x(t) and the new output as o(t+1) = -x(t).
    def call(self, x, curr_state):
        #After running this script, notice the following in the print:
        # * The dynamic_rnn changes the names of the x and curr_state.
        # * The shape of x would not include the sequence size as we only work with a single input item at a time.
        print('curr_state:', curr_state)
        print('x:', x)
        print()
        
        new_state = x
        new_output = -x
        return (new_output, new_state)

###################################
class Model(object):
    
    def __init__(self):
        num_inputs = 1
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            init_state = tf.constant([ 0, 0 ], tf.float32, [2], 'init')

            #The initial state must be defined as a batch of initial states for each item in the minibatch.
            #Use tf.tile to replicate the same initial state vector for each item in the batch.
            batch_init_states = tf.tile(tf.reshape(init_state, [1, 2]), [ num_inputs, 1 ])

            #Using the RNN cell for a single input-state combination.
            input_vec = tf.constant([1, 1], tf.float32, [2])
            c = Cell()
            (self.output_cell, self.state_cell) = c(input_vec, batch_init_states) #Just treat the cell as if it is a function.

            #Using the RNN cell to process a whole sequence of inputs.
            seqs = tf.constant(
                    [
                        [ [1, 1], [2, 2], [3, 3] ],
                    ], tf.float32, [1, 3, 2], 'seqs'
                )
            cell = Cell()
            #Note how the input shape given to the cell's call function is [1,2], i.e. a single item from every sequence in the batch.
            (self.outputs, self.state) = tf.nn.dynamic_rnn(cell, seqs, initial_state=batch_init_states)

            self.graph.finalize()

            self.sess = tf.Session()

    def close(self):
        self.sess.close()
    
    def output(self):
        return self.sess.run([ self.output_cell, self.state_cell, self.outputs, self.state ], {  })
    
###################################
model = Model()
[ output_cell, state_cell, outputs, state ] = model.output()
print('output_cell:', output_cell)
print('state_cell:', state_cell)
print('outputs:')
print(outputs)
print('state:', state)