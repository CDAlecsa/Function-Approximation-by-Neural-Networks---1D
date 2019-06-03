'''
                             Functions for approximation        
'''

from numpy import sin, cos, exp, pi


#%%%%%%%%%%%%%%%%%%%%%%%    Function with high oscillations    %%%%%%%%%%%%%%%%

def f1(x):
    return x * cos(71 * x) + sin(13 * x)

    
#%%%%%%%%%%%%%%%%%%%%%%%    Polynomial with sin and cos function  %%%%%%%%%%%%%

def f2(x):
    return 0.2 + 0.4 * x ** 2 + 0.3 * x * sin(15 * x) + 0.05 * cos(50 *x)
    

#%%%%%%%%%%%%%%%%%%%%%%%    Sine with exponential function    %%%%%%%%%%%%%%%%%
    
def f3(x):
    return sin(4 * pi * x) * exp(-abs(5*x))
    

#%%%%%%%%%%%%%%%%%%%%%%%    Exponential function    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
def f4(x):
    return (x + 1) * exp(-3 * x + 3)
    

#%%%%%%%%%%%%%%%%%%%%%%%    Piecewise function   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
def f5(x):
    if -10 <= x and x < -2 :
        return -2.186 * x - 12.864
    elif -2 <=x and x < 0 :
        return 4.264 * x
    elif 0 <= x and x <= 10 :
        return 10 * exp(-0.05 * x - 0.5) * sin(x * (0.03 * x + 0.7))
    
    
#%%%%%%%%%%%%%%%%%%%%%%%    Dictionary of functions   %%%%%%%%%%%%%%%%%%%%%%%%%
        
dict = {
        'high_osc' : f1,
        'pol_sin_cos' : f2,
        'exp_sin' : f3,
        'exp' : f4,
        'piecewise' : f5
        }    
    