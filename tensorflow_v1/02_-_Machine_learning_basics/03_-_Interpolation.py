import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

#The training set of points to interpolate.
points_x = [ -2.0, -1.0, 0.0, 1.0, 2.0 ]
points_y = [ 3.22, 1.64, 0.58, 1.25, 5.07 ]

###################################

class Model(object):

    def __init__(self, degree):
        learning_rate = 0.0005
        num_coefficients = degree + 1
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.xs = tf.placeholder(tf.float32, [None], 'xs') #The x values of the points.
            self.ts = tf.placeholder(tf.float32, [None], 'ts') #The y values of the points (called targets).

            self.coeffs = []
            for i in range(num_coefficients):
                coeff = tf.get_variable('coeff_'+str(i), [], tf.float32, tf.zeros_initializer())
                self.coeffs.append(coeff)
                if i == 0:
                    self.ys = coeff
                else:
                    self.ys = self.ys + coeff*self.xs**i
            
            self.error = tf.reduce_mean((self.ys - self.ts)**2)

            #Using Tensorflow's provided gradient descent function.
            self.optimiser_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.error)
            
            self.init = tf.global_variables_initializer()

            self.graph.finalize()
            
            self.sess = tf.Session()

    def initialise(self):
        self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self, xs, ts):
        return self.sess.run([ self.optimiser_step ], { self.xs: xs, self.ts: ts })
    
    def get_params(self):
        return self.sess.run(self.coeffs, { })
    
    def get_error(self, xs, ts):
        return self.sess.run([ self.error ], { self.xs: xs, self.ts: ts })[0]
    
    def predict(self, xs):
        return self.sess.run([ self.ys ], { self.xs: xs })[0]

###################################

max_epochs = 2000
degree = 6 #Create a polynomial of degree 6.

#Display figure at the start and update it during training.
(fig, axs) = plt.subplots(1, 2)

axs[0].plot(points_x, points_y, color='red', linestyle='', marker='o', markersize=10)
axs[0].set_title('Polynomial')
axs[0].set_xlim(-2.5, 2.5)
axs[0].set_xlabel('x')
axs[0].set_ylim(-10.0, 10.0)
axs[0].set_ylabel('y')
axs[0].grid(True)

[ error_plot ] = axs[1].plot([], [], color='red', linestyle='-', linewidth=1)
axs[1].set_title('Error progress')
axs[1].set_xlim(0, max_epochs)
axs[1].set_xlabel('epoch')
axs[1].set_ylim(0, 2)
axs[1].set_ylabel('MSE')
axs[1].grid(True)

fig.tight_layout()
fig.show()

xs = np.linspace(-2.5, 2.5, 30)

###################################

model = Model(degree)
model.initialise()

errors = list()
print('epoch', 'error', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', sep='\t')
for epoch in range(1, max_epochs+1):
    error = model.get_error(points_x, points_y)
    errors.append(error)
    
    if epoch%100 == 0: #Avoid displaying information about every single epoch by only doing so once every 100 epochs.
        coeffs = model.get_params()
        print(epoch, error, *coeffs, sep='\t')
        
        ys = model.predict(xs)
        axs[0].plot(xs, ys, color='magenta', linestyle='-', linewidth=1)
        error_plot.set_data(np.arange(len(errors)), errors)
        fig.canvas.draw()
        fig.canvas.flush_events()

    model.optimisation_step(points_x, points_y)

#Plot the final polynomial found in red.
ys = model.predict(xs)
axs[0].plot(xs, ys, color='red', linestyle='-', linewidth=3)
error_plot.set_data(np.arange(len(errors)), errors)
fig.canvas.draw()
fig.canvas.flush_events()

model.close()