'''
                             Activation functions           
'''

from numpy import exp, arctan, log, cos, sin




#%%%%%%%%%%%%%%%%%%%%%%%    Gaussian wavelet function    %%%%%%%%%%%%%%%%%%%%%%


def gaussian(z):
    return exp(- z ** 2 / 2)

def gaussian_derivative(z):
    return (-z) * exp(- z ** 2 / 2)




#%%%%%%%%%%%%%%%%%%%%%%%    Gaussian wavelet function    %%%%%%%%%%%%%%%%%%%%%%


def gaussian_wavelet(z):
    return (-z) * exp(- z ** 2 / 2)

def gaussian_wavelet_derivative(z):
    return (z ** 2 - 1) * exp(- z ** 2 / 2)




#%%%%%%%%%%%%%%%%%%%%%%%    Morlet function    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def morlet(z):
    return cos(5 * z) * exp(- z ** 2 / 2)

def morlet_derivative(z):
    return - (z * cos(5*z) + 5 * sin(5 * z)) * exp(- z ** 2 / 2)




#%%%%%%%%%%%%%%%%%%%%%%%    Identity function    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def identity(z):
    return z

def identity_derivative(z):
    return 1



#%%%%%%%%%%%%%%%%%%%%%%%    Sigmoid function    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
def sigmoid(z):
    return 1.0 / ( 1.0 + exp(-z) )

def sigmoid_derivative(z):
    return sigmoid(z) * ( 1 - sigmoid(z) )



#%%%%%%%%%%%%%%%%%%%%%%%    Hyperbolic tangent function    %%%%%%%%%%%%%%%%%%%%

def tanh(z):
    return 2 / ( 1 + exp(-2*z) ) - 1

def tanh_derivative(z):
    return 1 - tanh(z) ** 2



#%%%%%%%%%%%%%%%%%%%%%%%    Arctangent function    %%%%%%%%%%%%%%%%%%%%%%%%%%%%

def atan(z):
    return arctan(z)

def atan_derivative(z):
    return 1 / ( z ** 2 + 1 )

    
    
#%%%%%%%%%%%%%%%%%%%%%%%    Softplus function    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

def softplus(z):
    return log( 1 + exp(z) )

def softplus_derivative(z):
    return 1 / ( 1 + exp(-z) )
    


#%%%%%%%%%%%%%%%%%%%%%%%%  The functions are grouped in some dictionaries %%%%%


fct = {
        'identity' : identity,
        'sigmoid' : sigmoid,
        'tanh' : tanh,
        'atan' : atan,
        'softplus' : softplus,
        'gaussian' : gaussian,
        'gaussian_wavelet' : gaussian_wavelet,
        'morlet' : morlet,
        'mexican_hat' : mexican_hat
       }            


fct_deriv = {
        'identity_derivative' : identity_derivative,
        'sigmoid_derivative' : sigmoid_derivative,
        'tanh_derivative' : tanh_derivative, 
        'atan_derivative' : atan_derivative,
        'softplus_derivative' : softplus_derivative,
        'gaussian_derivative' : gaussian_derivative,
        'gaussian_wavelet_derivative' : gaussian_wavelet_derivative,
        'morlet_derivative' : morlet_derivative,
        'mexican_hat_derivative' : mexican_hat_derivative
       }   