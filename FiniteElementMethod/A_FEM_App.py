import numpy as np
import matplotlib.pyplot as plt
# Exact Solution of the differential equation -eps*y" + lamb*y' = f(x), y(0) = 0; y(1) = 0
def exact_solution(eps,lamb,fc,x):
    return (fc/lamb)*(x-((np.exp(lamb*x/eps)-1)/(np.exp(lamb/eps)-1)))
eps = 0.1 # the diffusivity constant
lamb = 1 #the speed of a fluid
f = 1 # for a constant function f
n = 20 # the number of intervals
h = 1/(n+1) 
# Computing B,C and A matrices
B = -np.eye(n,n,k=-1) + 2*np.eye(n,n) - np.eye(n,n,k=1)
C = -np.eye(n,n,k=-1) + np.eye(n,n,k=1)
A = eps*B/h + lamb*C/2
# Computing b
fc = f*np.ones(n)
b = h*fc
# Linear algebra part 
u = np.linalg.solve(A,b)
u = np.concatenate([[0],u,[0]])
x = np.linspace(0,1,len(u))
u_exact = exact_solution(eps,lamb,f,x)
# Plotting
plt.title("The solution of an Advection-Diffusion Equation by the FEM")
plt.plot(x,u,'ro--',label="the FEM Solution", linewidth=1, markersize=5)
plt.plot(x,u_exact,label="the Exact Solution",color = "b",linewidth = 2)
plt.legend()
plt.show()
