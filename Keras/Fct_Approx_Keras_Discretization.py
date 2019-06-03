'''
                                    Packages
'''

import itertools

import matplotlib.pyplot as plt

import Functions_Module as fcts

from numpy.random import uniform
from numpy import array, linspace, sort
from matplotlib.pyplot import show, plot, figure, legend, title, semilogy

import seaborn as sns
sns.set()


from keras import optimizers
from keras.layers import Dense, Activation
from keras.models import Sequential

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

'''
                                   New activation functions 
'''

def gaussian_wavelet(x):
    return (-x) * K.exp(- x ** 2 / 2)

def gaussian(x):
    return K.exp(- x ** 2 / 2)

def morlet(x):
    return K.cos(5*x) * K.exp(- x ** 2 / 2)

get_custom_objects().update({'morlet': Activation(morlet)})
get_custom_objects().update({'gaussian': Activation(gaussian)})
get_custom_objects().update({'gaussian_wavelet': Activation(gaussian_wavelet)})


'''
                                   Parameters
'''

epochs = 2000
learningRate = 1
momentum = 0.0
batchSize = 10


n_train_data = 1000
n_test_data = 200
parts = 4


'''
                                Given function for the approximation
'''


print('Choose between the following functions : [high_osc, pol_sin_cos, exp_sin, exp, piecewise]')
strFct = input('Choose function : ')

if strFct == 'high_osc' or strFct == 'pol_sin_cos' :
    low, high = 0, 1
    
elif strFct == 'exp_sin' :
    low, high = -1, 1
    
elif strFct == 'exp' :
    low, high = -2, 2
    
elif strFct == 'piecewise' :
    low, high = -10, 10
    




x_train = linspace(low, high, n_train_data)
x_test = uniform(low, high, n_test_data)
x_test = sort(x_test)


y_train = array([fcts.dict[strFct](x) for x in x_train])
y_test = array([fcts.dict[strFct](x) for x in x_test])



training_data = [(x,y) for x, y in zip(x_train, y_train)]
test_data = [(x,y) for x, y in zip(x_test, y_test)]



'''
                                Split the training and test datasets
'''

def split_list(lst, sublist_number=1):
    length = len(lst)
    return [ lst[i*length // sublist_number : (i+1)*length // sublist_number] 
                 for i in range(sublist_number) ]




X_train = split_list(x_train, sublist_number = parts)
X_test = split_list(x_test, sublist_number = parts)


Y_train_pred = [None] * parts
Y_test_pred = [None] * parts


'''
                                    Build Model
'''

for i in range(parts):
    

    def buildModel_DNN():
    
        fct_model = Sequential([
                Dense(64, input_dim = 1, activation = 'tanh'),
                Dense(64, activation = 'tanh'),
                Dense(64, activation = 'tanh'),
                Dense(1, activation = 'linear')
                ])
    
        DNN_optimizer = optimizers.Adadelta(lr=learningRate, rho = 0.9)
    
        fct_model.compile(loss='mean_squared_error', optimizer=DNN_optimizer, metrics=['mean_squared_error'])
    
        return fct_model


    DNN_model = buildModel_DNN()
    print(DNN_model.summary(), '\n\n')


    history_model = DNN_model.fit(x_train, y_train, epochs=epochs, batch_size=batchSize, validation_data=(x_test, y_test))


    Y_train_pred[i]  = DNN_model.predict(X_train[i])
    Y_test_pred[i]  = DNN_model.predict(X_test[i])

    Y_train_pred[i]  = Y_train_pred[i].reshape(1,-1)
    Y_train_pred[i]  = Y_train_pred[i][0]

    Y_test_pred[i]  = Y_test_pred[i].reshape(1,-1)
    Y_test_pred[i]  = Y_test_pred[i][0]

    '''
                                            Loss plot
    '''
    
    strTitle = ' Interval no. ' + str(i+1)
    
    plt.figure()
    plt.plot(history_model.history['loss'], 'bo-', label = 'Training Data')
    plt.plot(history_model.history['val_loss'], 'gs-', label = 'Test Data')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss : training vs. testing data' + strTitle)
    plt.legend()
    

Y_train_pred = array(list(itertools.chain(*Y_train_pred)))
Y_test_pred = array(list(itertools.chain(*Y_test_pred)))

    
    
'''
                                    Approximation results on train dataset
'''
figure()
plot(x_train, y_train, 'ro-', label='Target Label')
plot(x_train, Y_train_pred, 'b*-', label='DNN prediction')
title('Train dataset')
legend()


figure()
plot(x_test, y_test, 'ro-', label='Target Label')
plot(x_test, Y_test_pred, 'k*-', label='DNN prediction')
title('Test dataset')
legend()


'''
                                    Error between the true and the predicted values
'''
figure()
semilogy(x_train, abs(Y_train_pred-y_train), 'gv--')
title('Absolute error : train dataset')



figure()
semilogy(x_test, abs(Y_test_pred-y_test), 'yv--')
title('Absolute error : test dataset')
show()