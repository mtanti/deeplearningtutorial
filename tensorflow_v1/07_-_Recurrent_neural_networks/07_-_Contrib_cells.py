import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

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
            2,
        ], tf.int32, [1]
    )
    
    #Simple RNN with a layer size of 2 and a tanh activation function
    srnn_cell = tf.contrib.rnn.BasicRNNCell(2, tf.tanh)
    (srnn_outputs, srnn_state) = tf.nn.dynamic_rnn(srnn_cell, seqs, sequence_length=seq_len, initial_state=init_state)
    
    #Gated Recurrent Unit with a layer size of 2
    gru_cell = tf.contrib.rnn.GRUCell(2)
    (gru_outputs, gru_state) = tf.nn.dynamic_rnn(gru_cell, seqs, sequence_length=seq_len, initial_state=init_state)
    
    '''
    GRU specification:
    def build(self, input_shape):
        self.W_g_z = self.add_variable('W_g_z', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
        self.b_g_z = self.add_variable('b_g_z', [state_size], tf.float32, tf.zeros_initializer())
        self.W_g_r = self.add_variable('W_g_r', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
        self.b_g_r = self.add_variable('b_g_r', [state_size], tf.float32, tf.zeros_initializer())
        self.W_s = self.add_variable('W_s', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
        self.b_s = self.add_variable('b_s', [state_size], tf.float32, tf.zeros_initializer())
        self.built = True
    
    def call(self, x, curr_state):
        gate_input = tf.concat([ curr_state, x ], axis=1)
        g_z = tf.sigmoid(tf.matmul(gate_input, self.W_g_z) + self.b_g_z)
        g_r = tf.sigmoid(tf.matmul(gate_input, self.W_g_r) + self.b_g_r)
        state_input = tf.concat([ g_r*curr_state, x ], axis=1)
        new_state = g_z*tf.tanh(tf.matmul(state_input, self.W_s) + self.b_s) + (1 - g_z)*curr_state
        return (new_state, new_state)
    '''
    
    #Long short-term memory with a layer size of 2
    #LSTMs use two separate initial states which means that you need to initialise both states and also have two final states (although the 'h' state is what is usually used as a final state)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(2)
    (lstm_outputs, lstm_state) = tf.nn.dynamic_rnn(lstm_cell, seqs, sequence_length=seq_len, initial_state=tf.contrib.rnn.LSTMStateTuple(h=init_state, c=init_state))
    lstm_state_h = lstm_state.h
    lstm_state_c = lstm_state.c
    
    '''
    LSTM specification:
    def build(self, input_shape):
        self.W_g_i = self.add_variable('W_g_i', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
        self.b_g_i = self.add_variable('b_g_i', [state_size], tf.float32, tf.zeros_initializer())
        self.W_g_f = self.add_variable('W_g_f', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
        self.b_g_f = self.add_variable('b_g_f', [state_size], tf.float32, tf.zeros_initializer())
        self.W_g_o = self.add_variable('W_g_o', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
        self.b_g_o = self.add_variable('b_g_o', [state_size], tf.float32, tf.zeros_initializer())
        self.W_s = self.add_variable('W_s', [state_size+embedding_size, state_size], tf.float32, tf.random_normal_initializer(stddev=init_stddev))
        self.b_s = self.add_variable('b_s', [state_size], tf.float32, tf.zeros_initializer())
        self.built = True
    
    def call(self, x, curr_state):
        (curr_hidden, curr_cell) = curr_state
        gate_input = tf.concat([ curr_hidden, x ], axis=1)
        g_i = tf.sigmoid(tf.matmul(gate_input, self.W_g_i) + self.b_g_i)
        g_f = tf.sigmoid(tf.matmul(gate_input, self.W_g_f) + self.b_g_f)
        g_o = tf.sigmoid(tf.matmul(gate_input, self.W_g_o) + self.b_g_o)
        cell_input = tf.concat([ curr_hidden, x ], axis=1)
        new_cell = g_i*tf.tanh(tf.matmul(cell_input, self.W_s) + self.b_s) + g_f*curr_cell
        new_hidden = g_o*tf.tanh(new_cell)
        new_state = (new_hidden, new_cell)
        return (new_state, new_state)
    
    #The above implementation uses tuples instead of the LSTMTuple class, returns both hidden and cell states
    #Don't forget to change the state_size and output_size to return (state_size, state_size)
    '''
    
    init = tf.global_variables_initializer()
    
    g.finalize()

    with tf.Session() as s:
        s.run([ init ], { })
        
        [
            curr_srnn_outputs, curr_srnn_state,
            curr_gru_outputs, curr_gru_state,
            curr_lstm_outputs, curr_lstm_state_h, curr_lstm_state_c
        ] = s.run([
            srnn_outputs, srnn_state,
            gru_outputs, gru_state,
            lstm_outputs, lstm_state_h, lstm_state_c
        ], { })

        print('-------------')
        print('srnn_outputs')
        print(curr_srnn_outputs)
        print()
        
        print('srnn_state')
        print(curr_srnn_state)
        print()
        print()

        print('-------------')
        print('gru_outputs')
        print(curr_gru_outputs)
        print()
        
        print('gru_state')
        print(curr_gru_state)
        print()
        print()

        print('-------------')
        print('lstm_outputs')
        print(curr_lstm_outputs)
        print()
        
        print('lstm_state_h')
        print(curr_lstm_state_h)
        print()
        
        print('lstm_state_c')
        print(curr_lstm_state_c)
