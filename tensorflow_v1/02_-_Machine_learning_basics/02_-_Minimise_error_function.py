import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

max_epochs = 10

class Model(object):

    def __init__(self):
        learning_rate = 0.01
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.min_x = tf.get_variable('min_x', [], tf.float32, tf.constant_initializer(1))

            #The two intersecting polynomials
            self.y1 = -2 + -2*self.min_x + self.min_x**2
            self.y2 = -1 +  2*self.min_x + self.min_x**2
            
            #The square error function
            self.error = (self.y1 - self.y2)**2
            
            [ self.grad ] = tf.gradients([ self.error ], [ self.min_x ])
            
            self.optimiser_step = tf.assign(self.min_x, self.min_x - learning_rate*self.grad)
            
            self.init = tf.global_variables_initializer()

            self.graph.finalize()
            
            self.sess = tf.Session()

    def initialise(self):
        self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self):
        return self.sess.run([ self.optimiser_step ], { })
    
    def curr_min_x(self):
        return self.sess.run([ self.min_x, self.error ], { })
    
    def get_error(self, x):
        return self.sess.run([ self.error ], { self.min_x: x })[0]
    
    def predict_y1(self, x):
        return self.sess.run([ self.y1 ], { self.min_x: x })[0]
    
    def predict_y2(self, x):
        return self.sess.run([ self.y2 ], { self.min_x: x })[0]
    

model = Model()
model.initialise()

print('epoch', 'x', 'error', sep='\t')
min_xs = list()
min_es = list()
for epoch in range(1, max_epochs+1):
    [ min_x, min_e ] = model.curr_min_x()
    min_xs.append(min_x)
    min_es.append(min_e)
    print(epoch, min_x, min_e, sep='\t')

    model.optimisation_step()

xs = np.linspace(-2.0, 2.0, 20).tolist()
es =  [ model.get_error(x) for x in xs ]
y1s = [ model.predict_y1(x) for x in xs ]
y2s = [ model.predict_y2(x) for x in xs ]

(fig, axs) = plt.subplots(2, 1)

#Plot the polynomials.
axs[0].plot(xs, y1s, color='red', linestyle='-', linewidth=3)
axs[0].plot(xs, y2s, color='magenta', linestyle='-', linewidth=3)
axs[0].set_title('Polynomial')
axs[0].set_xlim(-2.0, 2.0)
axs[0].set_xlabel('x')
axs[0].set_ylim(-10.0, 10.0)
axs[0].set_ylabel('y')
axs[0].grid(True)

#Plot the error function.
axs[1].plot(xs, es, color='magenta', linestyle='-', linewidth=3)
axs[1].plot(min_xs, min_es, color='blue', linestyle='', marker='o', markersize=5)
axs[1].set_title('Error')
axs[1].set_xlim(-2.0, 2.0)
axs[1].set_xlabel('x')
axs[1].set_ylim(-2.0, 10.0)
axs[1].set_ylabel('error')
axs[1].grid(True)

fig.tight_layout()
fig.show()

model.close()