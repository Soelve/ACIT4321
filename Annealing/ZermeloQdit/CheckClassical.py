"""This script simulates a quantum annealing scheme in which 
a vessel travels accross a river in which the current varies.
It does so by discretizing the space into d points in the y
direction and n points in the x direction. Initial and final
points are fixed by boundary conditions.
Specifically, the scheme simulates n qudits, each of dimension d.
The boundary conditions are fixed in the implementation. The
cost in going from point (i,k) to (i+1, k_prime) is encoded into 
the Hamiltonian through nearest coupling interactions.
Iputs:
S(x,y) - the current model
Ly - extension of the grid in y direction, y \in [-Ly/2, +Ly/2]
d - the dimension of the qudits, must be odd.
n - the number of qudits.
The vessels velocity and the distance accross the river are taken
as units of their respective quantities. Also \hbar is one."""                                  
                                   


# Bibliotek
import numpy as np
#import matplotlib.pyplot as plt
# Local libraries
from Libraries.Hamiltonians import TimeCost_v2
from Libraries.Bookkeeping import Index2State

# Inputs
v = 1
D = 1


# The current model
def CurrentFunk(x,y):
    return 0.9*np.exp(-(x-D/np.pi)**2)

# Width of the grid in the y-direction
Ly = 0.25

# Dimension of the qudits
d = 5
# The number of qudits
n = 6

# Increments in x and y
dx = D/(n+1)
dy = Ly/(d-1)    

MinTime = 1e6

# Loop over all states
for k in range(d**n):
    State = Index2State(k,d,n)
    TimeCost_tot = 0
    # All intermediate steps
    for step in range(n-1):
        xi = (step+1)*dx
        yi = State[step]*dy - Ly/2
        xf = (step+2)*dx
        yf = State[step+1]*dy - Ly/2
        TimeCost_tot += TimeCost_v2(xi,yi,xf,yf,v,CurrentFunk)
        
    # First step
    TimeCost_tot += TimeCost_v2(0,0,dx,State[0]*dy-Ly/2,v,CurrentFunk) 
    # Last step
    TimeCost_tot += TimeCost_v2(D-dx,State[-1]*dy-Ly/2,D,0,v,CurrentFunk) 
    
    if TimeCost_tot < MinTime:
        MinTime = TimeCost_tot
        MinState = State
        
print(MinState)        
     
