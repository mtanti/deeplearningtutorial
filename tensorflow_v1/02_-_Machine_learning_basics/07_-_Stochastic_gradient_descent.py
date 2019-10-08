import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

train_x = [ -2.0, -1.0, 0.0, 1.0, 2.0 ]
train_y = [ 3.22, 1.64, 0.58, 1.25, 5.07 ]
test_x  = [ -1.5, -0.5, 0.5, 1.5 ]
test_y  = [ 2.38, 0.05, 0.47, 1.67 ]

###################################

class Model(object):

    def __init__(self, degree):
        learning_rate = 0.0001 #Smaller learning rate.
        num_coefficients = degree + 1
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.xs = tf.placeholder(tf.float32, [None], 'xs')
            self.ts = tf.placeholder(tf.float32, [None], 'ts')

            self.coeffs = []
            for i in range(num_coefficients):
                coeff = tf.get_variable('coeff_'+str(i), [], tf.float32, tf.zeros_initializer())
                self.coeffs.append(coeff)
                if i == 0:
                    self.ys = coeff
                else:
                    self.ys = self.ys + coeff*self.xs**i
            
            self.error = tf.reduce_mean((self.ys - self.ts)**2)

            self.optimiser_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.error)
            
            self.init = tf.global_variables_initializer()

            self.graph.finalize()
            
            self.sess = tf.Session()

    def initialise(self):
        return self.sess.run([ self.init ], { })
    
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
degree = 6
minibatch_size = 2 #The minibatch size specifies how many training set items to show the optimiser at once.

(fig, axs) = plt.subplots(1, 2)

axs[0].plot(train_x, train_y, color='red', linestyle='', marker='o', markersize=10, label='train')
axs[0].plot(test_x, test_y, color='orange', linestyle='', marker='o', markersize=10, label='test')
[ polynomial_plot ] = axs[0].plot([], [], color='magenta', linestyle='-', linewidth=1)
axs[0].set_title('Polynomial')
axs[0].set_xlim(-2.5, 2.5)
axs[0].set_xlabel('x')
axs[0].set_ylim(-10.0, 10.0)
axs[0].set_ylabel('y')
axs[0].grid(True)
axs[0].legend()

[ train_error_plot ] = axs[1].plot([], [], color='red', linestyle='-', linewidth=1, label='train')
[ test_error_plot ] = axs[1].plot([], [], color='orange', linestyle='-', linewidth=1, label='test')
axs[1].set_title('Error progress')
axs[1].set_xlim(0, max_epochs)
axs[1].set_xlabel('epoch')
axs[1].set_ylim(0, 2)
axs[1].set_ylabel('MSE')
axs[1].grid(True)
axs[1].legend()

fig.tight_layout()
fig.show()

xs = np.linspace(-2.5, 2.5, 30)

###################################

model = Model(degree)
model.initialise()

train_errors = list()
test_errors = list()
print('epoch', 'train_error', 'test_error', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(train_x, train_y)
    train_errors.append(train_error)
    test_error = model.get_error(test_x, test_y)
    test_errors.append(test_error)
    
    if epoch%100 == 0:
        print(epoch, train_error, test_error, sep='\t')
        
        ys = model.predict(xs)
        polynomial_plot.set_data(xs, ys)
        train_error_plot.set_data(np.arange(len(train_errors)), train_errors)
        test_error_plot.set_data(np.arange(len(test_errors)), test_errors)
        fig.canvas.draw()
        fig.canvas.flush_events()

    #Shuffle a list of indexes for every training item.
    indexes = np.arange(len(train_x))
    np.random.shuffle(indexes)
    #Apply gradient descent on groups of training items individually.
    for i in range(int(np.ceil(len(indexes)/minibatch_size))): #Take the ceiling of the number of minibatches needed to process all the training set.
        minibatch_indexes = indexes[i*minibatch_size:(i+1)*minibatch_size].tolist() #These are the training set indexes of the items in the current minibatch.
        model.optimisation_step([ train_x[j] for j in minibatch_indexes ], [ train_y[j] for j in minibatch_indexes ])

print()
print('Test error:', test_error)

model.close()