import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt

max_epochs = 2000
patience = 3 #The patience specifies how many epochs to allow the model to act badly before stopping training.
start_checking_early_stopping_epoch = 30 #This is to avoid using early stopping during the initial epochs as they are very noisy.

class Model(object):

    def __init__(self, degree):
        learning_rate = 0.0005
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
            self.saver = tf.train.Saver()

            self.graph.finalize()
            
            self.sess = tf.Session()

    def initialise(self):
        return self.sess.run([ self.init ], { })
    
    #Need to save the best parameters.
    def save(self, path):
        self.saver.save(self.sess, path)
    
    #And load the best parameters.
    def load(self, path):
        self.saver.restore(self.sess, path)
    
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
    
model = Model(6)
model.initialise()

#Use three datasets: one for training, one for validating, and one for testing.
train_x = [ -2.0, -1.0, 0.0, 1.0, 2.0 ]
train_y = [ 3.22, 1.64, 0.58, 1.25, 5.07 ]
val_x   = [ -1.75, -0.75, 0.25, 1.25 ]
val_y   = [ 3.03, 0.64, 0.46, 0.77 ]
test_x  = [ -1.5, -0.5, 0.5, 1.5 ]
test_y  = [ 2.38, 0.05, 0.47, 1.67 ]

(fig, axs) = plt.subplots(1, 2)

axs[0].plot(train_x, train_y, color='red', linestyle='', marker='o', markersize=10, label='train')
axs[0].plot(val_x, val_y, color='yellow', linestyle='', marker='o', markersize=10, label='val') #Draw the validation data points in yellow.
axs[0].plot(test_x, test_y, color='orange', linestyle='', marker='o', markersize=10, label='test')
axs[0].set_title('Polynomial')
axs[0].set_xlim(-2.5, 2.5)
axs[0].set_xlabel('x')
axs[0].set_ylim(-10.0, 10.0)
axs[0].set_ylabel('y')
axs[0].grid(True)
axs[0].legend()

axs[1].plot([ 0 ], [ 0 ], color='red', linestyle='-', linewidth=1, label='train')
axs[1].plot([ 0 ], [ 0 ], color='yellow', linestyle='-', linewidth=1, label='val')
axs[1].plot([ 0 ], [ 0 ], color='orange', linestyle='-', linewidth=1, label='test')
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
ys = model.predict(xs)

train_errors = list()
val_errors = list()
test_errors = list()
best_val_error = np.inf #Initialise the best error to be the worst possible error.
epochs_since_last_best_val_error = 0 #Initialise the number of bad epochs to 0.
print('epoch', 'train_error', 'val_error', 'test_error', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', sep='\t')
for epoch in range(1, max_epochs+1):
    train_error = model.get_error(train_x, train_y)
    train_errors.append(train_error)
    val_error = model.get_error(val_x, val_y)
    val_errors.append(val_error)
    test_error = model.get_error(test_x, test_y)
    test_errors.append(test_error)
    
    #Early epochs are a bit unstable in terms of validation error so we only check for overfitting once training progress becomes smooth.
    if epoch >= start_checking_early_stopping_epoch:
        #If the current validation error is the best then reset the bad epochs counter.
        if val_error < best_val_error:
            best_val_error = val_error
            epochs_since_last_best_val_error = 0
            model.save('./model') #Save parameters since they are the best found till now.
        #If not then increment the counter.
        else:
            epochs_since_last_best_val_error += 1

        #If it has been too many epochs since the last best validation error then stop training early.
        if epochs_since_last_best_val_error >= patience:
            break
    
    if epoch%100 == 0:
        coeffs = model.get_params()
        print(epoch, train_error, val_error, test_error, *coeffs, sep='\t')
        
        ys = model.predict(xs)
        axs[0].plot(xs, ys, color='magenta', linestyle='-', linewidth=1)
        axs[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', linewidth=1)
        axs[1].plot(np.arange(len(val_errors)), val_errors, color='yellow', linestyle='-', linewidth=1)
        axs[1].plot(np.arange(len(test_errors)), test_errors, color='orange', linestyle='-', linewidth=1)
        fig.canvas.draw()
        fig.canvas.flush_events()

    model.optimisation_step(train_x, train_y)

#Load the best model parameters found till now. We know that the saved parameters are the best because we were only saving when a model did better than the previous best.
model.load('./model')

ys = model.predict(xs)
test_error = model.get_error(test_x, test_y)
axs[0].plot(xs, ys, color='red', linestyle='-', linewidth=3)
axs[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', linewidth=1)
axs[1].plot(np.arange(len(val_errors)), val_errors, color='yellow', linestyle='-', linewidth=1)
axs[1].plot(np.arange(len(test_errors)), test_errors, color='orange', linestyle='-', linewidth=1)
axs[1].annotate('Test error: '+str(test_error), (0,0)) #Write the test error on the figure.
print()
print('Test error:', test_error)
fig.canvas.draw()
fig.canvas.flush_events()

model.close()