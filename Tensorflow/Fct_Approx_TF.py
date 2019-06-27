'''
                                    Load Modules
'''


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy.random import normal



'''
                                    Random samples for the test and validation datasets
'''

N = 2500
N_test = 1500

low = 0 
high = 4

value_perturbation_1 = normal(scale=0.5, size=(1))
value_perturbation_2 = normal(scale=0.1, size=(1))


'''
                                    Train, validation and test datasets
'''

X_train = np.linspace(low, high, N).reshape((N, 1))
Y_train = np.cos(X_train) * np.sin(X_train)


X_val = X_train + value_perturbation_1
X_val = X_val.reshape((N, 1))
Y_val = np.cos(X_val) * np.sin(X_val)


X_test = X_train[0:N_test] + value_perturbation_2
X_test = X_test.reshape((N_test, 1))
Y_test = np.cos(X_test) * np.sin(X_test)




'''
                                    Parameters for the neural network
'''


lr = 1
epochs = 1000
batch_size = 32
std_dev = 0.01



'''
                                   Parameters for the neural network structure
'''

n_input = 1
n_hidden_1 = 100
n_hidden_2 = 100
n_output = 1





'''
                                  Weights of the neural network
'''

weights = {'W1' : tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev = std_dev)),
           'W2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev = std_dev)),
           'W_Out' : tf.Variable(tf.random_normal([n_hidden_2, n_output], stddev = std_dev)),
        }


'''
                                  Biases of the neural network
'''

biases = {'b1' : tf.Variable(tf.random_normal([n_hidden_1], stddev = std_dev)),
           'b2' : tf.Variable(tf.random_normal([n_hidden_2], stddev = std_dev)),
           'b_Out' : tf.Variable(tf.random_normal([n_output], stddev = std_dev)),
        }



'''
                                 Layers of the neural network
'''

def neural_network(X):
    
    layer_1 = tf.add(tf.matmul(X, weights['W1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['W2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_output = tf.add(tf.matmul(layer_2, weights['W_Out']), biases['b_Out'])

    return layer_output




'''
                                Loss function for the neural network
'''

def loss_function(predicted_value, true_value):
    average_loss = tf.reduce_mean(tf.square(predicted_value - true_value))
    return average_loss



'''
                                Optimizer for the neural network
'''

def init_optimizer():
    optimizer = tf.train.AdadeltaOptimizer(learning_rate = lr)
    return optimizer



'''
                                Get the current batch used in the training process
'''
    
def get_batch(X, Y, iteration, batch_size):
    batch_X = X[(iteration * batch_size) : ((iteration + 1) * batch_size), :]
    batch_Y = Y[(iteration * batch_size) : ((iteration + 1) * batch_size), :]
    return batch_X, batch_Y



'''
                                Prediction on a new dataset
'''

def predict_test(X_new):
    predicted_value = neural_network(X_new)
    return predicted_value



'''
                                Train the neural network
'''

def train(X_train, X_val, X_test, Y_train, Y_val, Y_test, verbose=True):
    
    # Building the Tensorflow graph
    X = tf.placeholder('float', [None, n_input], name='X')
    Y = tf.placeholder('float', [None, n_output], name='Y')
    
    # Predicted value
    predicted_value = neural_network(X)
    
    # Actual value
    true_value = Y
    
    # Average loss
    average_loss = loss_function(predicted_value, true_value)
    
    # Create optimizer
    optimizer = init_optimizer().minimize(average_loss)
    
    # Compute validation loss
    validation_loss = loss_function(predicted_value, true_value)
    
    # Initialize all the global variables
    init = tf.global_variables_initializer()
    
    # Starting Tensorflow session for the computation
    with tf.Session() as sess:
        
        # Global variables hold actual values
        sess.run(init)
        
        # Loop over the number of epochs
        for epoch in range(epochs):
            
            epoch_loss = 0.0
            
            # The number of batches in the dataset
            n_batches = np.round(N / batch_size).astype(int)
            
            # Loop over the number of the batches
            for iteration in range(n_batches):
                
                batch_X, batch_Y = get_batch(X_train, Y_train, iteration, batch_size)

                # Dataflow of the computational graph
                
                _, batch_loss = sess.run([optimizer, average_loss], feed_dict = {X : batch_X, Y : batch_Y})
                
                # Sum up the total loss using the batch loss
                
                epoch_loss += batch_loss
            
            # Average epoch loss
            
            epoch_loss = epoch_loss / n_batches
            
            # Compute validation loss
            
            val_loss = sess.run(validation_loss, feed_dict = {X : X_val, Y : Y_val})
            
            # Display epoch and loss values
            
            if epoch % 10 == 0:
                print('epoch : {} .............. train_loss : {:.3E} ............... val_loss : {:.3E}'.format(
                    epoch, round(epoch_loss, 3), round(float(val_loss), 3)))
            
        # Return the value prediction on the test dataset
        
        return sess.run(tf.squeeze(predicted_value), {X : X_train}), sess.run(tf.squeeze(predicted_value), {X : X_val}), sess.run(tf.squeeze(predicted_value), {X : X_test})
        
        # Close session
        
        sess.close()
        

'''
                                    Main function
'''        


Y_pred_train, Y_pred_val, Y_pred_test = train(X_train, X_val, X_test, Y_train, Y_val, Y_test, verbose=True)
Y_pred_train = Y_pred_train.reshape((N,1))
Y_pred_val = Y_pred_val.reshape((N,1))
Y_pred_test = Y_pred_test.reshape((N_test,1))




'''
                                   Exact and numerical solutions on the train dataset
'''

plt.figure()
plt.plot(X_train, Y_train, 'ro-', label=' True value ')
plt.plot(X_train, Y_pred_train, 'bv-', label=' Predicted value ')
plt.legend()
plt.title('Approximation : Train data')


'''
                                   Exact and numerical solutions on the validation dataset 
'''

plt.figure()
plt.plot(X_val, Y_val, 'ro-', label=' True value ')
plt.plot(X_val, Y_pred_val, 'bv-', label=' Predicted value ')
plt.legend()
plt.title('Approximation : Validation data')

'''
                                   Exact and numerical solutions 
'''

plt.figure()
plt.plot(X_test, Y_test, 'ro-', label=' True value ')
plt.plot(X_test, Y_pred_test, 'bv-', label=' Predicted value ')
plt.legend()
plt.title('Approximation : Test data')





'''
                                    Absolute error on the train dataset
'''

plt.figure()
plt.plot(X_train, abs(Y_train-Y_pred_train), 'k--')
plt.title('Absolute error : Train data')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))


'''
                                    Absolute error on the validation dataset
'''

plt.figure()
plt.plot(X_val, abs(Y_val-Y_pred_val), 'k--')
plt.title('Absolute error : Validation data')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))


'''
                                    Absolute error on the test dataset
'''

plt.figure()
plt.plot(X_test, abs(Y_test-Y_pred_test), 'k--')
plt.title('Absolute error : Test data')
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
plt.show()