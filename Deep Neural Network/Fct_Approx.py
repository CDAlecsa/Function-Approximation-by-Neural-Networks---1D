'''
                                Import packages
'''


import DNN
import Plots_Module as plts
import Functions_Module as fcts


from numpy.random import uniform
from numpy import sin, cos, array, linspace, sort, pi, exp
from matplotlib.pyplot import show, plot, figure, legend, title, semilogy



'''
                                Parameters
''' 

n_train_data = 1000
n_test_data = 200

epochs = 2000
lr = 1e-01
batch_length=10
strOptim = 'Adam'

gamma = 0.9
beta_1 = 0.9
beta_2 = 0.999
rho = 0.9




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
                                    Neural network training
'''

metrics = ['train_loss', 'test_loss']
DNN = DNN.neural_network([1, 64, 64, 1])
train_loss, test_loss = DNN.train_network(training_data=training_data, test_data=test_data, 
                                                                   mini_batch_size=batch_length, strOptim=strOptim,
                                                                   epochs=epochs, learning_rate=lr, metrics=metrics, gamma=gamma)
'''
                                    Train and test predictions
'''

y_train_pred = array(DNN.predict(x_train)).reshape(-1,1)
y_train_pred = y_train_pred.reshape(1,-1)[0]

y_test_pred = array(DNN.predict(x_test)).reshape(-1,1)
y_test_pred = y_test_pred.reshape(1,-1)[0]

'''
                                    Decrease in the loss function
'''
strTitle = ' Whole interval '
plts.plot_loss(strTitle=strTitle, number_of_epochs=epochs, train_loss=train_loss, test_loss=test_loss)


'''
                                    Approximation results on train dataset
'''
figure()
plot(x_train, y_train, 'ro-', label='Target Label')
plot(x_train, y_train_pred, 'b*-', label='DNN prediction')
title('Train dataset')
legend()


figure()
plot(x_test, y_test, 'ro-', label='Target Label')
plot(x_test, y_test_pred, 'k*-', label='DNN prediction')
title('Test dataset')
legend()


'''
                                    Error between the true and the predicted values
'''
figure()
semilogy(x_train, abs(y_train_pred-y_train), 'gv--')
title('Absolute error : train dataset')



figure()
semilogy(x_test, abs(y_test_pred-y_test), 'yv--')
title('Absolute error : test dataset')
show()