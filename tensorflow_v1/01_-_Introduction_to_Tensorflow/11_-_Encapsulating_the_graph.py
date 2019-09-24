import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Model(object):

    def __init__(self, degree):
        num_coefficients = degree + 1 #For example, 1 + 2x + 3x^2 is a polynomial of degree 2 but with 3 coefficients.
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.float32, [], 'x')

            self.coeffs = [] #The list of coefficients in the polynomials.
            for i in range(num_coefficients):
                coeff = tf.get_variable('coeff_'+str(i), [], tf.float32, tf.random_normal_initializer()) #Create a coefficient variable.
                self.coeffs.append(coeff)
                
                #Add new term with created coefficient to the polynomial.
                if i == 0:
                    self.y = coeff #The constant term. Start the polynomial from the constant and then continue adding terms to it.
                else:
                    self.y = self.y + coeff*self.x**i

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

            self.graph.finalize()
            
            self.sess = tf.Session()

    def initialise(self):
        self.sess.run([ self.init ], { })
    
    def get_params(self):
        return self.sess.run(self.coeffs, { })
    
    def save(self, path):
        self.saver.save(self.sess, path)
    
    def load(self, path):
        self.saver.restore(self.sess, path)
    
    def close(self):
        self.sess.close() #This manually closes the session to release any resources it was using.
    
    def predict(self, x):
        return self.sess.run([ self.y ], { self.x: x })[0]

###################################

#Use the model.
model = Model(2) #Create a polynomial of degree 2 (a quadratic).
model.initialise() #Initialise the model's variables (coefficients).
print('f(0) =', model.predict(0.0)) #Get the polynomial output when the input is 0.
model.close() #Close the polynomial's session.