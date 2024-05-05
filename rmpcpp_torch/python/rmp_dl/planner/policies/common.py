import numpy as np  

def h(c, z):
    return z + c * np.log(1 + np.exp(- 2 * c * z))

def softmax(c, x):
    return x / h(c, np.linalg.norm(x))

class SimpleTarget:
    def __init__(self, alpha, beta, c_softmax):
        """Simple target from the RMP paper, but directly called with (x_g - x). 

        Args:
            alpha (_type_): Attractor gain
            beta (_type_): Velocity damping gain
            c_softmax (_type_): Softmax constant
        """
        self.alpha = alpha
        self.beta = beta
        self.c_softmax = c_softmax

    def __call__(self, x, v):
        return self.alpha * softmax(self.c_softmax, x) - self.beta * v
