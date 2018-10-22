import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

g = tf.Graph()
with g.as_default():
    xs = tf.placeholder(tf.float32, [None], 'xs')
    
    ys_sig = tf.sigmoid(xs)
    ys_tanh = tf.tanh(xs)
    ys_softmax = tf.nn.softmax(xs)
    
    g.finalize()

    with tf.Session() as s:
        (fig, ax) = plt.subplots(1, 4)
        
        all_xs = np.linspace(-10, 10, 50)
        [ all_ys ] = s.run([ ys_sig ], {xs: all_xs})
        
        ax[0].plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3)
        ax[0].set_xlim(-10, 10)
        ax[0].set_xlabel('x')
        ax[0].set_ylim(-1.1, 1.1)
        ax[0].set_ylabel('y')
        ax[0].set_title('sigmoid')
        ax[0].grid(True)

        all_xs = np.linspace(-10, 10, 50)
        [ all_ys ] = s.run([ ys_tanh ], {xs: all_xs})
        
        ax[1].plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3)
        ax[1].set_xlim(-10, 10)
        ax[1].set_xlabel('x')
        ax[1].set_ylim(-1.1, 1.1)
        ax[1].set_title('tanh')
        ax[1].grid(True)

        all_xs = [-1, 0, 1]
        [ all_ys ] = s.run([ ys_softmax ], {xs: all_xs})

        ax[2].bar(all_xs, all_ys)
        for (x, y) in zip(all_xs, np.round(all_ys, 2)):
            ax[2].annotate(y, xy=(x, y), xytext=(x, y+5), textcoords='offset points', ha='center', va='bottom')
        ax[2].set_xlabel('logits')
        ax[2].set_ylim(-1.1, 1.1)
        ax[2].set_title('softmax')
        ax[2].grid(True)

        all_xs = [-1-1, 0-1, 1-1]
        [ all_ys ] = s.run([ ys_softmax ], {xs: all_xs})

        ax[3].bar(all_xs, all_ys)
        for (x, y) in zip(all_xs, np.round(all_ys, 2)):
            ax[3].annotate(y, xy=(x, y), xytext=(x, y+5), textcoords='offset points', ha='center', va='bottom')
        ax[3].set_xlabel('logits')
        ax[3].set_ylim(-1.1, 1.1)
        ax[3].set_title('softmax')
        ax[3].grid(True)
        
        fig.tight_layout()
        fig.show()
