import math
import numpy as np 

"""
    This function calculates the roots of the quadratic inequality for the Rh reuse factor.
    Parameters:
        lx - list of input sizes of the lstms. The size of this list is equal to the number of layers.
        lh - list of input sizes of the hidden layers. The size of this list is equal to the number of layers.
        lt_sigma - the latency of the sigmoid/tanh functions.
        lt_tail - the latency of the tail.
        dsp_total - the total number of dsps
    This returns the roots of the quadratic inequality.
"""
def reuse_factor(lx, lh, lt_sigma, lt_tail, dsp_total):

    a = dsp_total - 4 * sum(lh)
    b = dsp_total * (lt_sigma + lt_tail) - 4 * np.dot(lx, lh) - 4 * np.dot(lh, lh) - 4 * (lt_sigma + lt_tail) * sum(lh)
    c = - 4 * (lt_sigma + lt_tail) * np.dot(lh, lh)
#    print(a)
#    print(b)
#    print(c)

    r_1 = (-b + math.sqrt(b**2 - 4*a*c)) / (2*a)
    r_2 = (-b - math.sqrt(b**2 - 4*a*c)) / (2*a)
    
    return r_1, r_2


print("ZYNQ")
print(reuse_factor([1,9],[9,9], 3,8,220))

print("lstm_ae_small exmaple")
print(reuse_factor([1,9],[9,9], 3,8,900))

print("\n")

print("KU115")
print("mnist 1/2 layers examples")
print(reuse_factor([28],[32], 3,8,5520))
print(reuse_factor([28,16],[16,16], 3,8,5520))

print("\n")
print("U250")
print("lstm_ae exmaple")
print(reuse_factor([1,32,8,8],[32,8,8,32], 3,8,12200))



