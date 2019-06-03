'''
                                Import packages
'''


import DNN, DNN_2
import Plots_Module as plts
import Functions_Module as fcts
import itertools


from numpy.random import uniform
from numpy import sin, cos, array, linspace, sort
from matplotlib.pyplot import show, plot, figure, legend, title, semilogy



'''
                                Parameters
''' 



n_train_data = 1000
n_test_data = 200
parts = 4

epochs = 2000
lr = 1e-02
batch_length=10
strOptim = 'Adam'


gamma = 0.9
beta_1 = 0.9
beta_2 = 0.999
rho = 0.9

# quadratic tanh tanh identity Adadelta and NaG with 1e-02 (good for batch length 2) 
# layers 64 64 (also with 3 layers : 64 64 8)



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

training_data_batches = split_list(training_data, sublist_number = parts)
test_data_batches = split_list(test_data, sublist_number = parts)




'''
                                    Neural network training
'''


Y_train_pred = [None] * parts
Y_test_pred = [None] * parts


for i in range(parts):
    metrics = ['train_loss', 'test_loss']
    
    if i == 0 :
        Fct_DNN = DNN.neural_network([1, 64, 64, 1])
        train_loss, test_loss = Fct_DNN.train_network(training_data=training_data_batches[i], test_data=test_data_batches[i], 
                                                                   mini_batch_size=batch_length, strOptim=strOptim,
                                                                   epochs=epochs, learning_rate=lr, metrics=metrics, gamma=gamma)
    
        
        new_weights = Fct_DNN.weights
        new_biases = Fct_DNN.biases
        
    else:
        Fct_DNN = DNN_2.neural_network_2([1, 64, 64, 1], new_weights, new_biases)
        train_loss, test_loss = Fct_DNN.train_network(training_data=training_data_batches[i], test_data=test_data_batches[i], 
                                                                   mini_batch_size=batch_length, strOptim=strOptim,
                                                                   epochs=epochs, learning_rate=lr, metrics=metrics, gamma=gamma)
    
        new_weights = Fct_DNN.weights
        new_biases = Fct_DNN.biases
    
    
    Y_train_pred[i] = array(Fct_DNN.predict(X_train[i])).reshape(-1,1)
    Y_train_pred[i] = Y_train_pred[i].reshape(1,-1)[0]

    Y_test_pred[i] = array(Fct_DNN.predict(X_test[i])).reshape(-1,1)
    Y_test_pred[i] = Y_test_pred[i].reshape(1,-1)[0]

    strTitle = ' Interval no. ' + str(i+1)
    plts.plot_loss(strTitle = strTitle, number_of_epochs=epochs, train_loss=train_loss, test_loss=test_loss)
        

    #plts.plot_loss(number_of_epochs=epochs, train_loss=train_loss, test_loss=test_loss)


'''
                                    Approximation results on train dataset
'''

Y_train_pred = array(list(itertools.chain(*Y_train_pred)))
Y_test_pred = array(list(itertools.chain(*Y_test_pred)))


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
