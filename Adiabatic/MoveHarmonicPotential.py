"""
 This script simulates a particle trapped in a harmonic potential
 which is shaken about a bit. Dependeing on the rate at which it is 
 shaken, the final state may or may not have a large overlap
 with the initial ground state. This overlap, the fidelity, is calculated
 in the end.
 
 Specifically, the harmonic oscillator potential is subject to a 
 time-dependent translation which shifts it in both directions before 
 restoring it in its original position. In addition to time, it takes 
 omega as an input; the duraton of the time-dependent shift is related 
 to omega by T = 2*pi/omega.

 Inputs
 L         - size of domain 
 N         - number of grid points (should be 2^n)
 Nt        - the number of time steps for each run
 k         - the strength of the potential
 omega     - the rate at which the potential is shaken
 Textra    - additional time after dynamics
 InitialState - the index of the initial state;  InitialState = 0 is 
 the ground state

 All inputs are hard coded initially
"""

# Libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg


# Grid parameters
L = 15
N = 512              

# Strength of the harmonic oscillator
Kpot = 1

# Time parameters
#Nstep = 500
dt = 0.025
omega = .5

# Extra time for splashing around afterwards
Textra = 7

# Index of initial state (ground state corresponds to zero)
InitialState = 0

# Shape of the potential
def Vpot(x):
    return 0.5*Kpot*x**2

# Duration of the simulation
Tshake = 2*np.pi/omega

# The displacement of the potential
def Ftrans(t, omega): 
    return (t<Tshake) * 8/3**(3/2)*np.sin(omega*t/2)**2*np.sin(omega*t)

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Kinetic energy operator (FFT)
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Transform identity matrix
Tmat = np.fft.fft(np.identity(N, dtype=complex), axis = 0)
# Multiply by (ik)^2
Tmat = np.matmul(np.diag(-k**2), Tmat)
# Transform back to x-representation. 
Tmat = np.fft.ifft(Tmat, axis = 0)
# Correct pre-factor (unit mass and \hbar = 1)
Tmat = -1/2*Tmat    

# Add potential
# Full Hamiltonian
Ham = Tmat + np.diag(Vpot(x))

# Diagaonalize time-independent Hamiltonian (Hermitian matrix)
Evector, Bmat = np.linalg.eigh(Ham)
# Normalize eigenstates
Bmat = Bmat/np.sqrt(h)

# InitialState
Psi0 = Bmat[:, InitialState]

# Initiate plots
plt.ion()
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot()
line1, = ax.plot(x, np.abs(Psi0)**2, '-', color='black')
line2, = ax.plot(x, np.abs(Psi0)**2, '--', color='blue', 
                 linewidth = 1)
# Scaling and plotting the potential
Psi0Max = np.max(np.abs(Psi0)**2)
line3, = ax.plot(x, Vpot(x), '-', color='red')
plt.xlabel('x')

# Fix window and wait for button
ax.set(ylim=(0, 1), xlim=(-5, 5))
plt.show(block=False)
plt.pause(0.1)
plt.waitforbuttonpress()
  
# Duration and time-step
#dt = Tshake/Nstep
# Propagator for half-step with kinetic energy
UkinHalf = linalg.expm(-1j*Tmat*dt/2)
t = 0
# Initial state
Psi = Psi0
 
# Do the dynamics
while t < Tshake + Textra:
   # Half step with kinetic energy
   Psi = np.matmul(UkinHalf, Psi)  
   # Full step with poetntial energy
   Translation = Ftrans(t+dt/2, omega)
   UpotFull = np.diag(np.exp(-1j*Vpot(x-Translation)*dt))
   Psi = np.matmul(UpotFull, Psi)
   # Another half step with kinetic energy
   Psi = np.matmul(UkinHalf, Psi)
   
   # Update plot
   line1.set_ydata(np.power(np.abs(Psi), 2))
   line3.set_ydata(Vpot(x-Translation))
   fig.canvas.draw()
   #fig.canvas.flush_events()
   plt.pause(0.015)

   # Update time
   t = t + dt

# Place initial state plot ontop of final state
line2, = ax.plot(x, np.abs(Psi0)**2, '--', color='blue', 
                 linewidth=1)

# Probability for remaining in the initial state
Pinit = np.abs(np.trapezoid(np.conj(Psi)*Psi0, dx=h))**2
# Write result to screen
print(f'omega: {omega:.4f}, Pinit: {Pinit:.4f}')
