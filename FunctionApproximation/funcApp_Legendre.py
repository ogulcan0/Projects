import numpy as np
from scipy.special import eval_legendre, roots_legendre
import matplotlib.pyplot as plt
# Legendre Polynomials
def Ln(x,n): # Easy of use "Legendre Polynomials"
    return eval_legendre(n,x)
def f(x): # the function to approximate
    return np.sin(6*x)*np.exp(-x)
X = np.linspace(-1, 1, 100) # The domain of the Legendre polynomials for "Orthogonality"
for p in range(6,10,3): # Legendre approximations for different p "degree of polynomials"
    roots, weights = roots_legendre(p) # Roots and weights of Legendre Polynomials
    coeff = np.zeros(p) # Computations of the Legendre coefficients
    for j in range(p):
        I = 0
        for i in range(len(weights)):
            I += f(roots[i])*Ln(roots[i],j)*weights[i]
        coeff[j] = (j+1/2) * I
    appr = np.zeros(len(X))
    for j in range(p):  # Computing approximation to the function
        appr += coeff[j]*Ln(X,j)
    plt.plot(X,appr,label = f"L{p} Approximation",marker = "*")
plt.plot(X,f(X),label = "The function",linewidth = 2,color = "black")
plt.title("An approximation of functions by Legendre Polynomials")
plt.legend()
plt.xlim(-1, 1)
plt.show()
