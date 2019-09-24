import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

max_epochs = 10 #Train for a maximum of 10 epochs.

class Model(object):

    def __init__(self):
        learning_rate = 0.2 #Use a learning rate of 0.2.

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.min_x = tf.get_variable('min_x', [], tf.float32, tf.constant_initializer(1))

            self.y = self.min_x**2
            
            [ self.grad ] = tf.gradients([ self.y ], [ self.min_x ])
            
            #Gradient descent equation for a single step.
            self.optimiser_step = tf.assign(self.min_x, self.min_x - learning_rate*self.grad)
            
            self.init = tf.global_variables_initializer()

            self.graph.finalize()
            
            self.sess = tf.Session()

    def initialise(self):
        self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self):
        return self.sess.run([ self.optimiser_step ], { }) #Apply a single optimisation step.
    
    def curr_min_x(self):
        return self.sess.run([ self.min_x, self.y ], { }) #Get the current min_x and its associated y value.
    
    def predict(self, x):
        return self.sess.run([ self.y ], { self.min_x: x })[0]
    
###################################

model = Model()
model.initialise()

#Find where each new min_x lands on the graph.
print('epoch', 'x', 'y', sep='\t')
min_xs = list()
min_ys = list()
for epoch in range(1, max_epochs+1): #Optimize min_x for 10 times (epochs).
    [ min_x, min_y ] = model.curr_min_x()
    min_xs.append(min_x)
    min_ys.append(min_y)
    print(epoch, min_x, min_y, sep='\t')

    #Optimize min_x a little here.
    model.optimisation_step()

xs = np.linspace(-2.0, 2.0, 20).tolist() #Get all values between -2 and 2 divided into 20 steps.
ys = [ model.predict(x) for x in xs ]

(fig, ax) = plt.subplots(1, 1)

ax.plot(xs, ys, color='red', linestyle='-', linewidth=3)
ax.plot(min_xs, min_ys, color='blue', linestyle='', marker='o', markersize=5) #Show a point for each min_x update.
ax.set_title('Polynomial')
ax.set_xlim(-2.0, 2.0)
ax.set_xlabel('x')
ax.set_ylim(-10.0, 10.0)
ax.set_ylabel('y')
ax.grid(True)

fig.tight_layout()
fig.show()

model.close()