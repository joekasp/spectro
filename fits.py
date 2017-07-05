import numpy as np

def my_exponential(t,a,b,c):
    return b*(1-np.exp(-a*t)) + c


def my_sine(t,a,b,c):
    return b*np.sin(a*t) + c


