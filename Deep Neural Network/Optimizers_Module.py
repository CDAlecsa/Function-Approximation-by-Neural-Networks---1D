'''
                           Stochastic Optimizers 
          (the input learning rate contains the size of the mini-batches)
'''




#%%%%%%%%%%%%%%%%%   Stochastic Gradient Descent   %%%%%%%%%%%%%%%%%%%%%%%%%%%%
def SGD(X, grad_X, lr):                           
  sol = [x - lr * grad for x, grad in zip(X, grad_X)]
  return sol


#%%%%%%%%%%%%%%%%%          Momentum & NaG       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def Momentum(X, grad_X, vel_X, lr, gamma=0.9):
    vel_X = [gamma * v + lr * grad for v, grad in zip(vel_X, grad_X)]
    sol = [x - v for x, v in zip(X, vel_X)]
    return vel_X, sol



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%        RMS Prop           %%%%%%%%%%%%%%%%%%%%%%%
def RMSprop(X, grad_X, E_X, lr, gamma):        
  E_X = [gamma * e + (1-gamma) * grad ** 2 for e, grad in zip(E_X, grad_X)]  
  sol = [x - (lr/( 1e-8 + e ) ** (1/2)) * grad for x, e, grad in zip(X, E_X, grad_X)]
  return E_X, sol


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%        Adam           %%%%%%%%%%%%%%%%%%%%%%%%%%%
def Adam(it, X, grad_X, E_m_X, E_v_X, lr, beta_1, beta_2):
    E_m_X = [beta_1 * e + (1-beta_1) * grad for e, grad in zip(E_m_X, grad_X)]
    E_v_X = [beta_2 * e + (1-beta_2) * grad ** 2 for e, grad in zip(E_v_X, grad_X)]
    
    m_hat = [ m / ( 1 - beta_1 ** it ) for m in E_m_X ]
    v_hat = [ v / ( 1 - beta_2 ** it ) for v in E_v_X ]
    
    sol = [ x - ( lr / ( 1e-8 + v ** (1/2) ) ) * m  for x, v, m in zip(X, v_hat, m_hat) ]
    
    return E_m_X, E_v_X, sol


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%        Adadelta        %%%%%%%%%%%%%%%%%%%%%%%%%%
def Adadelta(X, grad_X, E_grad, E_X, RMS_X, rho):
    E_grad = [ rho * e + (1-rho) * grad ** 2 for e, grad in zip(E_grad, grad_X) ]
    RMS_grad = [ ( 1e-8 + e ) ** (1/2) for e in E_grad ]
    delta_X = [ - (e1/e2) * grad for e1, e2, grad in zip(RMS_X,RMS_grad,grad_X)]
    E_X = [ rho * e + (1-rho) * dx ** 2 for e, dx in zip(E_X, delta_X) ]    
    RMS_X = [ ( 1e-8 + e ) ** (1/2) for e in E_X ]
    sol = [x + dx for x, dx in zip(X, delta_X)]     
    return E_grad, E_X, RMS_X, sol    





