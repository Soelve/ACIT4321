"""
This script simulates a spin 1/2-particle which is exposed to a magnetic
field. This field has both a static part and a dynamic part. The dynamic part
corresponds to a single cycle of a sine-like function.
The static field points in the z-direction, thus lifting the degenerecy
between spin up and spin down. The oscillating field is taken to point along
the x-axis, corresponding to a real coupling in the Hamiltonian.
The initial state is a spin up-state.
The implementation solves the Schrödinger equation (TDSE) by using the
SciPy function solve_ivp. It determines the probability to remain in the
spin up-state after interaction for various values of the angular frequency
omega.
The inputs are
Eps - the energy separation induced by the static field
W - the strength of the oscillating field
omega_min - minimal omega
omega_max - maximum omega
N_omega - the number of omega values to impose
All inputs are hard coded initially.
"""

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
