import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import skopt

tf.logging.set_verbosity(tf.logging.ERROR)

max_epochs = 2000
start_checking_early_stopping_epoch = 30

num_random_hyperparams = 10 #The number of random hyperparameters to evaluate at the beginning
num_chosen_hyperparams = 10 #The number of hyperparameters to try during the tuning phase

#Validation set was split into two: one for hyperparameter tuning and one for early stopping
train_x = [ -2.0, -1.0, 0.0, 1.0, 2.0 ]
train_y = [ 3.22, 1.64, 0.58, 1.25, 5.07 ]
val_x   = [ -1.75, 0.25 ]
val_y   = [ 3.03, 0.46 ]
tuneval_x   = [ -0.75, 1.25 ]
tuneval_y   = [ 0.64, 0.77 ]
test_x  = [ -1.5, -0.5, 0.5, 1.5 ]
test_y  = [ 2.38, 0.05, 0.47, 1.67 ]

#Since we will need to run different versions of model using different hyperparameters, we put the whole define-train-evaluate process in a function that takes in hyperparameters as parameters
#We also specify whether the function is being used for hyperparameter searching or for the actual training phase by specifying an extra parameter to avoid doing what's unnecessary during tuning
def train_model(learning_rate, patience, weight_decay_weight, minibatch_size, momentum, initialiser_stddev, for_hyperparameter_search=False):
    g = tf.Graph()
    with g.as_default():
        xs = tf.placeholder(tf.float32, [None], 'xs')
        ts = tf.placeholder(tf.float32, [None], 'ts')

        c0 = tf.get_variable('c0', [], tf.float32, tf.zeros_initializer())
        c1 = tf.get_variable('c1', [], tf.float32, tf.random_normal_initializer(stddev=initialiser_stddev))
        c2 = tf.get_variable('c2', [], tf.float32, tf.random_normal_initializer(stddev=initialiser_stddev))
        c3 = tf.get_variable('c3', [], tf.float32, tf.random_normal_initializer(stddev=initialiser_stddev))
        c4 = tf.get_variable('c4', [], tf.float32, tf.random_normal_initializer(stddev=initialiser_stddev))
        c5 = tf.get_variable('c5', [], tf.float32, tf.random_normal_initializer(stddev=initialiser_stddev))
        c6 = tf.get_variable('c6', [], tf.float32, tf.random_normal_initializer(stddev=initialiser_stddev))
        
        ys = c0 + c1*xs + c2*xs**2 + c3*xs**3 + c4*xs**4 + c5*xs**5 + c6*xs**6
        
        error = tf.reduce_mean((ys - ts)**2)
        params_size = c1**2 + c2**2 + c3**2 + c4**2 + c5**2 + c6**2
        loss = error + weight_decay_weight*params_size

        step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

        init = tf.global_variables_initializer()
        
        g.finalize()

        with tf.Session() as s:
            s.run([ init ], { })

            if not for_hyperparameter_search:
                (fig, ax) = plt.subplots(1, 2)
                plt.ion()
            
                train_errors = list()
                val_errors = list()
                
            best_val_error = np.inf
            epochs_since_last_best_val_error = 0
            if not for_hyperparameter_search:
                print('epoch', 'trainerror', 'valerror', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', sep='\t')
            for epoch in range(1, max_epochs+1):
                indexes = np.arange(len(train_x))
                np.random.shuffle(indexes)
                for i in range(int(np.ceil(len(indexes)/minibatch_size))):
                    minibatch_indexes = indexes[i*minibatch_size:(i+1)*minibatch_size]
                    s.run([ step ], { xs: [ train_x[j] for j in minibatch_indexes ], ts: [ train_y[j] for j in minibatch_indexes ] })

                if not for_hyperparameter_search:
                    [ curr_c0, curr_c1, curr_c2, curr_c3, curr_c4, curr_c5, curr_c6 ] = s.run([ c0, c1, c2, c3, c4, c5, c6 ], { })
                    [ train_error ] = s.run([ error ], { xs: train_x, ts: train_y })
                [ val_error ]  = s.run([ error ], { xs: val_x,  ts: val_y })
                if not for_hyperparameter_search:
                    train_errors.append(train_error)
                    val_errors.append(val_error)
                else:
                    #If the validation error is invalid then stop everything and return an invalid value
                    if np.isnan(val_error) or np.isinf(val_error):
                        return np.nan

                if epoch > start_checking_early_stopping_epoch:
                    if val_error < best_val_error:
                        best_val_error = val_error
                        epochs_since_last_best_val_error = 0
                    else:
                        epochs_since_last_best_val_error += 1

                    if epochs_since_last_best_val_error >= patience:
                        break

                if not for_hyperparameter_search and epoch%100 == 0:
                    print(epoch, train_error, val_error, round(curr_c0, 3), round(curr_c1, 3), round(curr_c2, 3), round(curr_c3, 3), round(curr_c4, 3), round(curr_c5, 3), round(curr_c6, 3), sep='\t')
                    
                    ax[0].cla()
                    ax[1].cla()

                    all_xs = np.linspace(-2.5, 2.5, 30)
                    [ all_ys ] = s.run([ ys ], { xs: all_xs })
                    ax[0].plot(all_xs, all_ys, color='blue', linestyle='-', linewidth=3)
                    ax[0].plot(train_x, train_y, color='red', linestyle='', marker='o', markersize=10, label='train')
                    ax[0].plot(val_x, val_y, color='yellow', linestyle='', marker='o', markersize=10, label='val')
                    ax[0].plot(test_x, test_y, color='orange', linestyle='', marker='o', markersize=10, label='test')
                    ax[0].set_xlim(-2.5, 2.5)
                    ax[0].set_xlabel('x')
                    ax[0].set_ylim(-10.0, 10.0)
                    ax[0].set_ylabel('y')
                    ax[0].set_title('Polynomial')
                    ax[0].grid(True)
                    ax[0].legend()

                    ax[1].plot(np.arange(len(train_errors)), train_errors, color='red', linestyle='-', label='train')
                    ax[1].plot(np.arange(len(val_errors)), val_errors, color='yellow', linestyle='-', label='val')
                    ax[1].set_xlim(0, max_epochs)
                    ax[1].set_xlabel('epoch')
                    ax[1].set_ylim(0, 1)
                    ax[1].set_ylabel('MSE')
                    ax[1].grid(True)
                    ax[1].set_title('Error progress')
                    ax[1].legend()
                    
                    fig.tight_layout()
                    plt.draw()
                    plt.pause(0.0001)

            if not for_hyperparameter_search:
                [ test_error ]  = s.run([ error ], { xs: test_x,  ts: test_y })
                ax[1].annotate('Test error: '+str(test_error), (0,0))
                print('Test error:', test_error)
            
                fig.show()
            else:
                [ test_error ]  = s.run([ error ], { xs: tuneval_x,  ts: tuneval_y })
            
            #Return the test error for use by the hyperparameter optimiser
            return test_error

opt = skopt.Optimizer(
    [
        skopt.space.Real(1e-6, 1e-1, 'log-uniform', name='learning_rate'), #The learning rate can be any real number between 1e-6 and 1e-1 on a log scaled distribution in order to choose a variety of exponent values (otherwise most values will not have any zeros after the point)
        skopt.space.Integer(10, 100, name='patience'), #The patience can be any integer between 10 and 100
        skopt.space.Real(1e-3, 1e-1, 'log-uniform', name='weight_decay_weight'),
        skopt.space.Integer(1, 5, name='minibatch_size'),
        skopt.space.Real(1e-2, 1e-0, 'log-uniform', name='momentum'),
        skopt.space.Real(1e-5, 1e-3, 'log-uniform', name='initialiser_stddev'),
    ],
    n_initial_points=num_random_hyperparams, #The number of random hyperparameters to try initially for the algorithm to have a feel of where to search
    base_estimator='RF', #Use Random Forests to predict which hyperparameters will result in good training
    acq_func='EI', #Choose a set of hyperparameters that maximise the Expected Improvement
    acq_optimizer='auto', #Let algorithm figure out how to find the most promising hyperparameters to try next
    random_state=0,
)
#There's also skopt.space.Categorical([val1, val2, val3], name='categorical_hyperparam') for when you want to choose a value from a given list
#You can leave out 'log-uniform' if you're not looking for exponential values
#See https://scikit-optimize.github.io/#skopt.Optimizer for information on the Optimizer function
#See https://scikit-optimize.github.io/space/space.m.html for information on each optimisable data type

print('Starting hyperparameter tuning')
print()
print('#', 'learning_rate', 'patience', 'weight_decay_weight', 'minibatch_size', 'momentum', 'initialiser_stddev', 'error', sep='\t')

best_hyperparams = None
best_error = np.inf
for i in range(1, num_random_hyperparams + num_chosen_hyperparams + 1):
    if i == 1:
        print('starting random search phase')
    if i == num_random_hyperparams+1:
        print('starting baysian optimisation phase')
    
    num_hyperpars = 1
    while True:
        #Since some hyperparameters might lead to errors, we need a way to be able to ask the optimiser for a new hyperparameter combination, test them, then only accept them if they are successful, otherwise ignore them and ask for different ones
        #This is achieved by a system of 'ask' and 'tell' in skopt
        
        #Ask for one hyperparameter combination, if it was good then ask for two next time and take the last, and so on (ask gives a list of hyperparameter candidates and will return the same candidates if you ask for the same number of candidates)
        next_hyperparams = opt.ask(num_hyperpars)[-1]
        (learning_rate, patience, weight_decay_weight, minibatch_size, momentum, initialiser_stddev) = next_hyperparams
        
        print(i, learning_rate, patience, weight_decay_weight, minibatch_size, momentum, initialiser_stddev, sep='\t', end='\t')
        
        #Test the hyperparameters by attempting to get the error
        try:
            error = train_model(learning_rate, patience, weight_decay_weight, minibatch_size, momentum, initialiser_stddev, for_hyperparameter_search=True)
            if np.isnan(error) or np.isinf(error):
                raise ValueError()
        except ValueError: #If some kind of error happend during evaluation of the model or if the error returned was NaN or infinite, ask for another candidate hyperparameter combination
            print('error, retrying')
            num_hyperpars += 1
            continue
        
        print(error)
        
        #Once a proper hyperparameter combination has been found, tell the optimiser about it together with its associated error in order to give it more information about which hyperparameter to suggest next
        #Unfortunately skopt can only be used to maximise not minimise so we have to negate the error in order to make it maximise the negative error
        opt.tell(next_hyperparams, -error)
        
        if error < best_error:
            best_hyperparams = next_hyperparams
            best_error = error
            
        break #Stop the while loop

print()
print('Tuning finished, starting actual model training')
print('Best hyperparameters found:')
print(best_hyperparams)
print()

#Finally take the best found hyperparameters and use them
(learning_rate, patience, weight_decay_weight, minibatch_size, momentum, initialiser_stddev) = best_hyperparams
train_model(learning_rate, patience, weight_decay_weight, minibatch_size, momentum, initialiser_stddev, for_hyperparameter_search=False)