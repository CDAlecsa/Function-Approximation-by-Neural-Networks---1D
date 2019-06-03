'''
            Plot functions for the accuracy and for the loss function
'''


from numpy import arange
import matplotlib.pyplot as plt




#%%%%%%%%%%%%%%%%%%%%%%%    Loss plot   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

def plot_loss(strTitle, number_of_epochs, train_loss, test_loss=[]):
    plt.figure()
    
    
    plt.semilogy(arange(1, number_of_epochs+1), train_loss[0:number_of_epochs], 'ro-', label='train_loss')
    if test_loss != []:
        plt.semilogy(arange(1, number_of_epochs+1), test_loss[0:number_of_epochs], 'bs-', label='val_loss')
    
    plt.xlim([0, number_of_epochs])       
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs' + strTitle)
    plt.legend()  