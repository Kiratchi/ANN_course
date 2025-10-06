import numpy as np

# SOURCES
# Introduction to Numpy https://numpy.org/doc/stable/user/absolute_beginners.html
# Funtion to create a binary array: https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding 
# Sampling bernoulli distribution: https://stackoverflow.com/questions/19597473/binary-random-array-with-a-specific-proportion-of-ones
# Sampling normal distribution: https://numpy.org/doc/2.1/reference/random/generated/numpy.random.normal.html
# Syntax help in python: ChatGPT


def sgn(x):
    return 1 if x >= 0 else -1


# Function taken from https://stackoverflow.com/questions/22227595/convert-integer-to-binary-array-with-suitable-padding
def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)
        


def determine_linearity(test_bolean, n, max_epoch = 20, eta = 0.05):
    
    w = np.random.normal(0,1/n,n) 
    theta = 0
    
    for current_epoch in range(max_epoch):
        for mu in range(2**n):
            x = bin_array(mu,n)
            t = test_bolean[mu] 
            O = sgn(w @ x.T - theta)

            dw = eta * (t - O) * x.T
            dtheta = - eta * (t - O)

            w += dw
            theta += dtheta

    # Checking the pattern is seperable
    separable = True
    for mu in range(2**n):
        x = bin_array(mu,n)
        t = test_bolean[mu] 
        O = sgn(w @ x.T - theta)

        if (O!= t):
            separable = False
            break
    return(separable)


def estimate_separability_frac(n):
    separable_count = 0
    if n < 4:
        total = 2**(2**n)
        for bool_idx in range(total):
            test_boolean = 2*bin_array(bool_idx, 2**n) -1
            separable_count += determine_linearity(test_boolean, n)
    else:
        total = 10**4
        for sample_idx in range(total):
                    test_boolean = 2*np.random.binomial(1,0.5, size=2**n) -1
                    separable_count += determine_linearity(test_boolean, n)

    return separable_count #/ total


print(estimate_separability_frac(2))
print(estimate_separability_frac(3))
print(estimate_separability_frac(4))
print(estimate_separability_frac(5))


