import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import skopt

#Use a validation set to evaluate models whilst optimising hyperparameters and use test set to evaluate final model obtained.
train_x = [ -2.0, -1.0, 0.0, 1.0, 2.0 ]
train_y = [ 3.22, 1.64, 0.58, 1.25, 5.07 ]
val_x   = [ -1.75, -0.75, 0.25, 1.25 ]
val_y   = [ 3.03, 0.64, 0.46, 0.77 ]
test_x  = [ -1.5, -0.5, 0.5, 1.5 ]
test_y  = [ 2.38, 0.05, 0.47, 1.67 ]

###################################

num_random_hyperparams = 10 #The number of random hyperparameters to evaluate during the exploration phase.
num_chosen_hyperparams = 20 #The number of selected hyperparameters to evaluate during the tuning phase.

#The hyperparameter optimiser.
opt = skopt.Optimizer(
    [
        #The learning rate can be any real number between 1e-6 and 0.1 on a log scaled distribution in order to choose a variety of exponent values (otherwise most values will not have any zeros after the point).
        skopt.space.Real(1e-6, 1e-1, 'log-uniform', name='learning_rate'),
        
        #The momentum can be any real number between 1e-2 and 1.0 on a log scaled distribution.
        skopt.space.Real(1e-2, 1e-0, 'log-uniform', name='momentum'),
        
        #The maximum number of epochs can be any integer number between 1 and 2000 both inclusive.
        skopt.space.Integer(1, 2000, name='max_epochs'),
    ],
    n_initial_points=num_random_hyperparams, #The number of random hyperparameters to try initially for the algorithm to have a feel of where to search.
    base_estimator='RF', #Use Random Forests to predict which hyperparameters will result in good training.
    acq_func='EI', #Choose a set of hyperparameters that maximise the Expected Improvement of the model.
    acq_optimizer='auto', #Let algorithm figure out how to find the most promising hyperparameters to try next.
)
#There's also skopt.space.Categorical([value1, value2, value3], name='categorical_hyperparam') for when you want to choose a value from a given list.
#You can leave out 'log-uniform' if you're not looking for exponential values.
#See https://scikit-optimize.github.io/#skopt.Optimizer for information on the Optimizer class.
#See https://scikit-optimize.github.io/space/space.m.html for information on each optimisable data type.

###################################

class Model(object):

    def __init__(self, degree, learning_rate, momentum, max_epochs): #Hyperparameters are parameterised. Note that since we also want to optimise hyperparameters that control the training algorithm, that is, the maximum number of epochs, then we will have to add a method for completely training the model here.
        self.max_epochs = max_epochs
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

            self.optimiser_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(self.error)
            
            self.init = tf.global_variables_initializer()

            self.graph.finalize()
            
            self.sess = tf.Session()

    def initialise(self):
        return self.sess.run([ self.init ], { })
    
    def close(self):
        self.sess.close()
    
    def optimisation_step(self, xs, ts):
        return self.sess.run([ self.optimiser_step ], { self.xs: xs, self.ts: ts })
    
    def train(self, xs, ts): #Completely train the model from scratch.
        self.initialise()
        for epoch in range(1, self.max_epochs+1):
            self.optimisation_step(train_x, train_y)
    
    def get_params(self):
        return self.sess.run(self.coeffs, { })
    
    def get_error(self, xs, ts):
        return self.sess.run([ self.error ], { self.xs: xs, self.ts: ts })[0]
    
    def predict(self, xs):
        return self.sess.run([ self.ys ], { self.xs: xs })[0]

###################################

degree = 6

print('Starting hyperparameter tuning')
print()
print('#', 'learning_rate', 'momentum', 'max_epochs', 'error', sep='\t')

#Note that the below code is the long but controllable version of using skopt. For the single line code version you can follow this link:
#https://scikit-optimize.github.io/optimizer/index.html#skopt.optimizer.forest_minimize

best_hyperparams = None
best_error = np.inf
for i in range(1, num_random_hyperparams + num_chosen_hyperparams + 1):
    if i == 1:
        print('Starting random search phase')
    if i == num_random_hyperparams+1:
        print('Starting baysian optimisation phase')
    
    num_hyperparams_to_ask = 1
    while True:
        #Since some hyperparameters might lead to errors, we need a way to be able to ask the optimiser for a new hyperparameter combination, test them, then only accept them if they are successful, otherwise ignore them and ask for different ones. Unfortunately this process is not automatic in skopt but it does let you ask for an amount of different hyperparameter combinations and then let you tell it which one you picked.
        
        #Ask for one hyperparameter combination. If it was a bad combination then ask for two next time and take the last, and so on until one that works is found. The ask method gives a list of hyperparameter candidates and will return the same candidates if you ask for the same number of candidates.
        new_hyperparams = opt.ask(num_hyperparams_to_ask)[-1]
        (learning_rate, momentum, max_epochs) = new_hyperparams

        #Test the hyperparameters by attempting to get the error.
        try:
            model = Model(degree, learning_rate, momentum, max_epochs)
            model.train(train_x, train_y)
            val_error = model.get_error(val_x, val_y)
            model.close()

            if np.isnan(val_error) or np.isinf(val_error):
                raise ValueError()
        except ValueError: #If some kind of error happened during evaluation of the model or if the error returned was NaN or infinite, ask for another candidate hyperparameter combination.
            print(i, learning_rate, momentum, max_epochs, 'error, retrying', sep='\t')
            num_hyperparams_to_ask += 1
            continue
        
        #If we get to here then we have found a valid hyperparameter combination and can continue.
        print(i, learning_rate, momentum, max_epochs, val_error, sep='\t')
        break
    
    #Once a proper hyperparameter combination has been found, tell the optimiser about it together with its associated error in order to give it more information about which hyperparameter to suggest next. Unfortunately skopt can only be used to maximise not minimise so we have to make it maximise the negative error.
    opt.tell(new_hyperparams, -val_error)
    
    #If a new best hyperparameter combination was found then keep it.
    if val_error < best_error:
        best_hyperparams = new_hyperparams
        best_error = val_error

print()
print('Tuning finished')
print()

(learning_rate, momentum, max_epochs) = best_hyperparams
print('Best hyperparameters found:')
print('learning_rate:', learning_rate)
print('momentum:', momentum)
print('max_epochs:', max_epochs)
print()
print('Starting actual model training')
print()

#Finally take the best found hyperparameters and use them.
model = Model(degree, learning_rate, momentum, max_epochs)
model.train(train_x, train_y)
test_error = model.get_error(test_x, test_y)

print('Test error:', test_error)

model.close()