import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None], 'xs')
    
    #The 3 of the most common activation functions
    ys_sig = tf.sigmoid(xs)
    ys_tanh = tf.tanh(xs)
    ys_relu = tf.nn.relu(xs)
    ys_softmax = tf.nn.softmax(xs)
    
    g.finalize()

    with tf.Session() as s:
        (fig, ax) = plt.subplots(1, 5)
        
        #Sigmoid
        all_xs = np.linspace(-10, 10, 50)
        [ all_ys ] = s.run([ ys_sig ], { xs: all_xs })
        
        ax[0].plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3)
        ax[0].set_xlim(-10, 10)
        ax[0].set_xlabel('x')
        ax[0].set_ylim(-1.1, 1.1)
        ax[0].set_ylabel('y')
        ax[0].set_title('sigmoid')
        ax[0].grid(True)

        #Hyperbolic tangent
        all_xs = np.linspace(-10, 10, 50)
        [ all_ys ] = s.run([ ys_tanh ], { xs: all_xs })
        
        ax[1].plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3)
        ax[1].set_xlim(-10, 10)
        ax[1].set_xlabel('x')
        ax[1].set_ylim(-1.1, 1.1)
        ax[1].set_title('tanh')
        ax[1].grid(True)

        #Rectified linear unit
        all_xs = np.linspace(-10, 10, 50)
        [ all_ys ] = s.run([ ys_relu ], { xs: all_xs })
        
        ax[2].plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3)
        ax[2].set_xlim(-10, 10)
        ax[2].set_xlabel('x')
        ax[2].set_ylim(-1.1, 1.1)
        ax[2].set_title('ReLU')
        ax[2].grid(True)

        #Softmax
        all_xs = [ -1, 0, 1 ]
        [ all_ys ] = s.run([ ys_softmax ], { xs: all_xs })

        ax[3].bar(all_xs, all_ys)
        for (x, y) in zip(all_xs, np.round(all_ys, 2)):
            ax[3].annotate(y, xy=(x, y), xytext=(x, y+5), textcoords='offset points', ha='center', va='bottom')
        ax[3].set_xlabel('logits')
        ax[3].set_ylim(-1.1, 1.1)
        ax[3].set_title('softmax')
        ax[3].grid(True)

        #Softmax with added constant to logits (xs)
        #Adding -1 to every element in the logits will not change the output probabilities.
        all_xs = [ -1 + -1, 0 + -1, 1 + -1 ]
        [ all_ys ] = s.run([ ys_softmax ], { xs: all_xs })

        ax[4].bar(all_xs, all_ys)
        for (x, y) in zip(all_xs, np.round(all_ys, 2)):
            ax[4].annotate(y, xy=(x, y), xytext=(x, y+5), textcoords='offset points', ha='center', va='bottom')
        ax[4].set_xlabel('logits')
        ax[4].set_ylim(-1.1, 1.1)
        ax[4].set_title('softmax')
        ax[4].grid(True)
        
        fig.tight_layout()
        fig.show()