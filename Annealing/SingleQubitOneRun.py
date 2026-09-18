"""
This script simulates a spin 1/2-particle which is exposed to a magnetic 
field which more or less slowly goes from pointning in the negative 
x-direction to the positive z-direction.

The initial state is a spin right-state, i.e., the ground state of the
initial Hamiltonian, H_i = -\sigma_x.

The implementation solves the Schrödinger equation (TDSE) by using the 
SciPy function solve_ivp. It plots the  probability of a spin down-measurement
and a spin right measurement as functions of time. I.e., it plots the 
ground state probabilities for both the initial and the final Hamiltonians. 

The only input is the duration of the proces, which is called Tfinal.
The larger Tfinal is, the more adiabatic is the process.
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

# Duration of the simulation
Tfinal = 5

# Set up the equation the right hand side of the ODE,
# y'(t) = -i H(t) y(t), where y is the spinor and H is
# the Hamiltonian
# Initial Hamiltonian
Hi = -np.matrix([[0, 1], [1, 0]])
# Final Hamiltonian
Hf = np.matrix([[1, 0], [0, -1]])
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

# Extract vector with times and amplitudes
t_vector = sol.t
a_vector = sol.y[0, :]       
b_vector = sol.y[1, :]       
# Spin up-probability
spin_left_prob = np.abs(a_vector)**2

# Calculate projection onto right eigenstate
proj_vector_right = 1/np.sqrt(2) * (a_vector + b_vector)

# Plot result
plt.figure(1)
plt.clf()
plt.plot(t_vector, np.abs(proj_vector_right)**2, 'b-', 
         label = 'Spin right-probability')
plt.plot(t_vector, np.abs(b_vector)**2, 'r--', 
         label = 'Spin down-probability')
plt.plot(t_vector, s(t_vector), 
         'k:', label = 'Schedule function')
plt.grid()
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Probability', fontsize = 12)
plt.legend(loc='lower right')
#plt.ylim(0, 1.1)
plt.show()

# Determine "fidelity", i.e., the probability to end up in the spin down state
Fid = np.abs(b_vector[-1])**2
print(f'Fidelity: {Fid*100:.2f}%')
