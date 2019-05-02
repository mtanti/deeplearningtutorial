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
    
    #Long short-term memory with a layer size of 2
    #LSTMs use two separate initial states which means that you need to initialise both states and also have two final states (although the 'h' state is what is usually used as a final state)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(2)
    (lstm_outputs, lstm_state) = tf.nn.dynamic_rnn(lstm_cell, seqs, sequence_length=seq_len, initial_state=tf.contrib.rnn.LSTMStateTuple(h=init_state, c=init_state))
    lstm_state_h = lstm_state.h
    lstm_state_c = lstm_state.c
    
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