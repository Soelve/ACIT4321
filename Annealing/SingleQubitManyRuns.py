"""
This script simulates a spin 1/2-particle which is exposed to a magnetic 
field which more or less slowly goes from pointning in the negative 
x-direction to the positive z-direction.

The initial state is a spin right-state, i.e., the ground state of the
initial Hamiltonian, H_i = -\sigma_x.

The implementation solves the Schrödinger equation (TDSE) by using the 
SciPy function solve_ivp. 

It solves the evolution for a number of durations T. In the end, it plots 
the probability for ending up in the final ground state as a function of T.

The inputs are the shortest T, the longest T and the steps size in T."""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

# Upper and lower limit for Tfinal
Tfinal_min = .1
Tfinal_max = 10
# The number of points
N_Tf = 100

# Vector with Tfinal values
Tf_vector = np.linspace(Tfinal_min, Tfinal_max, N_Tf)
# Allocate vector with spin up-probabilities
SpinDownProb_vector = np.zeros_like(Tf_vector)

# Initial Hamiltonian
Hi = -np.matrix([[0, 1], [1, 0]])
# Final Hamiltonian
Hf = np.matrix([[1, 0], [0, -1]])

index = 0
for Tfinal in Tf_vector:
     # Scheduling function
     def s(t):
         return 0.5*(1-np.cos(np.pi*t/Tfinal))
    
     def RHS(t, y):
       Ham = (1-s(t))*Hi + s(t)*Hf 
       Yderiv = np.matmul(Ham, y.reshape(2,1))
       return -1j*Yderiv
    
     # Initial state (must be  complex)
     y0 = 1/np.sqrt(2) * np.array([complex(1,0), complex(1, 0)])
    
     # ODE solver
     # In order to fix the time points for the output
     tVect = np.linspace(0, Tfinal, 500)
     # Numerical solution of the ODE
     sol = integrate.solve_ivp(RHS, (0, Tfinal), y0, t_eval = tVect, 
                               vectorized = True, rtol = 1e-6)
    
     # Extract final spin down-probability
     # Spin up-probability
     Pdown = np.abs(sol.y[1,-1])**2
     SpinDownProb_vector[index] = Pdown 
    
     index = index + 1

# Plott result
plt.figure(2)
plt.clf()
plt.plot(Tf_vector, 100*SpinDownProb_vector, 'k-')
plt.ylim((0, 110))
plt.grid(visible = True)
plt.xlabel('$T_f$')
plt.ylabel('Fidelity [%]')
plt.show()
