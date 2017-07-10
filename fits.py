import numpy as np

def my_exponential(t,a,b,c):
    return a*np.exp(-b*t) + c


def my_double_exp(t, a1, a2, b1, b2, c):
    return a1*np.exp(-b1*t) + a2*np.exp(-b2*t) + c


def my_sine(t,a,b,c):
    return a*np.sin(b*t) + c


