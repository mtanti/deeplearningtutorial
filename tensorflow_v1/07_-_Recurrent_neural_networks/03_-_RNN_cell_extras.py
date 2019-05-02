import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


class Cell(tf.nn.rnn_cell.RNNCell):

    def __init__(self):
        super().__init__()

    @property
    def state_size(self):
        return (2, 2) #Specify the vector size of each vector included in the state

    @property
    def output_size(self):
        return (2, 2) #Specify the vector size of each vector included in the outputs

    def build(self, inputs_shape):
        pass

    def call(self, x, curr_state):
        (curr_state1, curr_state2) = curr_state
        new_state1 = curr_state1 + x
        new_state2 = curr_state2 + new_state1
        new_output1 = new_state1
        new_output2 = new_state2
        return ((new_output1, new_output2), (new_state1, new_state2)) #Return all the vectors in a tuple


g = tf.Graph()
with g.as_default():
    #You can use separate initial states for each state vector
    init_state1 = tf.constant(
        [
            [ 0, 0 ],
        ], tf.float32, [1, 2], 'init1'
    )
    init_state2 = tf.constant(
        [
            [ 0, 0 ],
        ], tf.float32, [1, 2], 'init2'
    )
    init_state = (init_state1, init_state2) #Put initial states in a tuple

    seqs = tf.constant(
        [
            [ [1, 1], [2, 2], [3, 3] ],
        ], tf.float32, [1, 3, 2], 'seqs'
    )
    cell = Cell()
    (outputs, state) = tf.nn.dynamic_rnn(cell, seqs, initial_state=init_state)
    ((outputs1, outputs2), (state1, state2)) = (outputs, state) #You will receive a separate matrix for each output and a separate vector for each state

    g.finalize()

    with tf.Session() as s:
        [ curr_outputs1, curr_outputs2, curr_state1, curr_state2 ] = s.run([ outputs1, outputs2, state1, state2 ], { })

        print('outputs1:')
        print(curr_outputs1)
        print()
        
        print('outputs2:')
        print(curr_outputs2)
        print()

        print('state1:')
        print(curr_state1)
        print()
        
        print('state2:')
        print(curr_state2)
