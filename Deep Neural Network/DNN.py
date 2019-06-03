import numpy as np
import scipy.stats as scst
import Optimizers_Module as Optim
import Activations_Module as Activ


from numpy.linalg import norm


'''
                                Quadratic Loss Function
'''
def QUADRATIC_loss(a, y):  
    return 0.5 * norm(a-y) ** 2


def QUADRATIC_loss_derivative(a, y):
    return a - y


def QUADRATIC_loss_error(z, a, y, activ_deriv) :
    return QUADRATIC_loss_derivative(a, y) * activ_deriv(z)




'''
        Truncated normal distribution for the weights initialization, 
        using the interval (-1/sqrt(n), 1/sqrt(n)), 
        where n = the number of nodes in the input layer
'''
def TND(mean = 0.0, std = 1.0, low = 0.0, high = 10.0):
  a = ( low - mean ) / std
  b = ( high - mean ) / std
  return scst.truncnorm(a, b, loc = mean, scale = std)





'''
                            A single feedforward step
'''
# a = activation output in the L layer using the activation output from the L-1 layer

def feedforward_phase(b, w, a, fct_activ):
  z = np.dot(w, a) + b
  a = fct_activ(z)       
  return z, a




'''
                            The DNN class
'''
class neural_network():

  def __init__(self, layer_sizes):
    self.number_of_layers = len(layer_sizes)
    self.layer_sizes = layer_sizes

    self.weights_initializer()
    self.biases_initializer()
    self.params = [self.weights, self.biases]

    self.iteration = 1
    
    self.network_structure()


#%%%%%%%%%%%%%%%%%%%%%%%   Bias initialization from the 2nd layer  %%%%%%%%%%%%
    
  def biases_initializer(self):
    self.biases = np.array([np.random.randn(y, 1) for y in self.layer_sizes[1:]])




#%%%%%%%%%%%%   Weight initialization between consecutive layers    %%%%%%%%%%%
# Order of indices for w(j,k). Example : k = 1st layer and j = 2nd layers
    
  def weights_initializer(self):
    self.layer_array = np.linspace(1,self.number_of_layers, self.number_of_layers)
    sqrt_weight = [1 / np.sqrt(i) for i in self.layer_array[:-1]]
    self.weights = np.array([TND(mean = 0.0, std = 0.1, low = - value, high = value).rvs([y, x]) for value, x, y in zip(sqrt_weight, self.layer_sizes[:-1], self.layer_sizes[1:])])




#%%%%%%%%%%%%  Neural Network Structure   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  def network_structure(self):
      self.activ_list = []
      self.activ_list_derivative = []
      
      self.strLoss = 'QUADRATIC'
      
      
      
      print('Give the activation functions : ')
      for i in np.arange(2, self.number_of_layers):
         print(' The activation function between layers {} and {} .......... '.format(i-1 ,i)) 
         
         self.activ_list.append(input(''))
         self.activ_list[-1] = ''.join((self.activ_list[-1]).split()).lower()
         
         self.activ_list_derivative.append(self.activ_list[-1] + '_derivative')
         print('\n\n')
                
      self.activ_list.append('identity')
      self.activ_list_derivative.append(self.activ_list[-1] + '_derivative')
      print(' The activation function between layers {} and {} .......... '.format(self.number_of_layers - 1 ,self.number_of_layers))    
         
         

#%%%%%%%%%%     Print the shape of the weight matrices   %%%%%%%%%%%%%%%%%%%%%%
         
  def weights_print(self):
    print('\n ........................................................ \n')
    for i in range(self.number_of_layers-1):
      print('Weights between layer {} and layer {} : {}'.format(i+1, i+2, self.weights[i].shape))
    print('\n ........................................................... \n')  



#%%%%%%%%%%     Print the network structure  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
  def network_structure_print(self):
      print('Number of layers : ', self.number_of_layers, '\n')
      print('Number of activation functions : ', self.number_of_layers-1, '\n')
      print('The loss function :', self.strLoss, '\n')
      print('The activation functions :', self.activ_list, '\n\n')
      print('...............................................................', '\n\n')        
        

#%%%%%%%%%%  Update the list of parameters  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
      
  def update_params(self):
    self.weights = self.params[0] 
    self.biases = self.params[1]       
    


#%%%%  Partition the training data into mini-batches of a given size  %%%%%%%%%
    
  def batch_partitioning(self, training_data, mini_batch_size):
    n = len(training_data)
    np.random.shuffle(training_data)
    mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]
    return n, mini_batches



#%%%%%%%%%%%%%%%%%%    Initialize the optimizer parameters    %%%%%%%%%%%%%%%%%
    
  def initialize_optimizer(self, strOptim):
      
      if strOptim == 'Momentum' or strOptim == 'NaG':
          self.momentum_params = [[np.zeros(w.shape) for w in self.params[0]], [np.zeros(b.shape) for b in self.params[1]]]
          self.momentum_params = [np.array(i) for i in self.momentum_params]
          
          
      if strOptim == 'RMSprop':
          self.energy_params = [[0 for w in self.params[0]], [0 for b in self.params[1]]]
          self.energy_params = [np.array(i) for i in self.energy_params]
          
          
      if strOptim == 'Adam':
          self.m_params = [[0 for w in self.params[0]], [0 for b in self.params[1]]]
          self.m_params = [np.array(i) for i in self.m_params]
          
          self.v_params = [[0 for w in self.params[0]], [0 for b in self.params[1]]]
          self.v_params = [np.array(i) for i in self.v_params]
          
      if strOptim == 'Adadelta':
          
          self.E_delta_params = [[0 for w in self.params[0]], [0 for b in self.params[1]]]
          self.E_delta_params = [np.array(i) for i in self.E_delta_params]
          
          self.E_grad_params = [[0 for w in self.params[0]], [0 for b in self.params[1]]]
          self.E_grad_params = [np.array(i) for i in self.E_grad_params]
          
          self.RMS_params = [[0 for w in self.params[0]], [0 for b in self.params[1]]]
          self.RMS_params = [np.array(i) for i in self.RMS_params]
          
          
      
#%%%%%%%%%%%%%%%%%%    Run the chosen optimizer    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
          
  def run_optimizer(self, strOptim, gradient_params, learning_rate, m, gamma, beta_1, beta_2, rho):
      
      if strOptim == 'SGD':
          self.params = Optim.SGD(self.params, gradient_params, learning_rate / m)
          
      elif strOptim == 'Momentum' or strOptim == 'NaG':
          self.momentum_params, self.params = Optim.Momentum(self.params, gradient_params, self.momentum_params, learning_rate / m, gamma=gamma)
          
      elif strOptim == 'RMSprop':
          self.energy_params, self.params = Optim.RMSprop(self.params, gradient_params, self.energy_params, learning_rate / m, gamma=gamma)    
          
      elif strOptim == 'Adam':
          self.m_params, self.v_params, self.params = Optim.Adam(self.iteration, self.params, gradient_params, self.m_params, self.v_params, learning_rate / m, beta_1=beta_1, beta_2=beta_2)    
        
      elif strOptim == 'Adadelta':
          self.E_grad_params, self.E_delta_params, self.RMS_params, self.params = Optim.Adadelta(self.params, gradient_params, self.E_grad_params, self.E_delta_params, self.RMS_params, rho=rho)      
          



#%%%%%%%%%%%%%%%%%%%%%%%%%   Training the neural network     %%%%%%%%%%%%%%%%%%
          
  def train_network(self, training_data=None, test_data=None, strOptim='Momentum', gamma=0.0, beta_1 = 0.9, beta_2 = 0.999, rho=0.9, mini_batch_size=10, epochs=1, learning_rate=0.1, metrics=[]):
     
      
      self.train_loss, self.test_loss = [], []
      
      self.initialize_optimizer(strOptim)
      
      
      
      print('Optimizer : ', strOptim, '\n\n')
      self.weights_print()
      self.network_structure_print()
      print('Training the neural network \n\n\n')
      
      
      for i in range(epochs):
          
          n, mini_batches = self.batch_partitioning(training_data, mini_batch_size)
          for MB in mini_batches:
              self.update_mini_batch(i+1, MB, strOptim, learning_rate, n, gamma, beta_1, beta_2, rho)

    
          if 'train_loss' in metrics:
              T_loss = self.evaluate_loss(training_data)
              self.train_loss.append(T_loss)
              print('Epoch : {} _______________________ Train loss : {} '.format(i+1, T_loss), '\n')
            
          if 'test_loss' in metrics and test_data != None:
              V_loss = self.evaluate_loss(test_data)
              self.test_loss.append(V_loss)
              print('Epoch : {} _______________________ Test loss : {} '.format(i+1, V_loss), '\n')
          
          print("===========================================================")  
        
      return self.train_loss, self.test_loss
        


#%%%%%%%%%  For a given mini-batch, update the weights and biases  %%%%%%%%%%%%
      
  def update_mini_batch(self, epoch, MB, strOptim, learning_rate, n, gamma, beta_1, beta_2, rho):
    
    gradient_params = [[np.zeros(w.shape) for w in self.weights], [np.zeros(b.shape) for b in self.biases]]
    

    for x, y in MB:
      m = len(MB)
      
      error_params = self.backpropagation_algorithm(x, y, strOptim, gamma)
      
      gradient_params = [ [np + dnp for np, dnp in zip(gradient_params[i], error_params[i])] for i in range(0,2) ]
      


    gradient_params = [np.array(i) for i in gradient_params]
    
    self.run_optimizer(strOptim, gradient_params, learning_rate, m, gamma, beta_1, beta_2, rho)
    self.iteration += 1

    
    self.update_params()
    
    
    


#%%%%%%%%%%%%%%%%%%%%       Backpropagation Algorithm      %%%%%%%%%%%%%%%%%%%%
    
  def backpropagation_algorithm(self, x, y, strOptim, gamma):
        
    if strOptim == 'NaG':
        alg_params = [theta - gamma * v for theta, v in zip(self.params, self.momentum_params)] 
    else:
        alg_params = self.params
    
    error_params = [[np.zeros(w.shape) for w in alg_params[0]], [np.zeros(b.shape) for b in alg_params[1]]]
    

    current_activation, activation_list, weighted_input_list = x, [], []    # for a given (x, y) training element, we store weighted inputs and activation outputs layer by layer
    activation_list.append(current_activation)                              # the first activation output is the training example
    

    # FeedForward Phase (layer by layer)
    # in the forward phase we start from the 2nd layer and change the activation functions throughout this phase
    i = 0
    for b, w in zip(alg_params[1], alg_params[0]):
        z, current_activation = feedforward_phase(b, w, current_activation, Activ.fct[self.activ_list[i]])
        weighted_input_list.append(z)
        activation_list.append(current_activation)
        i += 1
        
           

    # Compute the error from the last layer
    # take the activation function beetwen the last 2 layers
    error = QUADRATIC_loss_error(weighted_input_list[-1], activation_list[-1], y, Activ.fct_deriv[self.activ_list_derivative[-1]])
    
    error_params[0][-1], error_params[1][-1] = [np.dot(error, activation_list[-2].transpose()), error]
    
    
    # Compute the errors starting from the second to last layer (i.e. backpropagate the error through the network)
    # start from the activation function between the last two layers and go through the list
    for L in range(2, self.number_of_layers):

      z = weighted_input_list[-L]
      zp = Activ.fct_deriv[self.activ_list_derivative[-L]](z)

      error = np.dot(alg_params[0][-L+1].transpose(), error) * zp
      
      error_params[0][-L], error_params[1][-L] = [np.dot(error, activation_list[-L-1].transpose()), error]
      
     
    return error_params


#%%%%%%%%%%%%%%%%%    Predict a result of the test data set    %%%%%%%%%%%%%%%%
    
  def run_network(self, a):
      i = 0
      for b, w in zip(self.params[1], self.params[0]):
          z = np.dot(w, a) + b
          a = Activ.fct[self.activ_list[i]](z)
          i += 1
      return a



#%%%%%%%%%%%%%%%%%%%%%%%    Return prediction   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      
  def predict(self, X_data):
      output_list = []
      for i in range(len(X_data)):
          x = X_data[i]
          output_list.append(self.run_network(x))
      return output_list


  
#%%%%%%%%%%%%%%%%%%%%%%%  Loss evaluation   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

  def evaluate_loss(self, data):
      loss = 0.0
      for x, y in data:
          a = self.run_network(x)
          loss += QUADRATIC_loss(a, y) / len(data)
      return loss    

