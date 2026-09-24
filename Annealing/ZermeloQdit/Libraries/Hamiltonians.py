"""This file contains functions to implement the various Hamiltonians
involved in the anneal."""

import numpy as np
# Functions which translates between global index and dit-string
from Libraries.Bookkeeping import Index2State, State2Index

#
# Functions which calculate the cost associated with specific steps
# 


# Calculate crossing time from one point to next
def TimeCost(xi,yi,Ystep,dx,dy,v,CurrentFunk):
    Slope = Ystep*dy/dx
    Ymean = yi + Ystep*dy/2
    Xmean = xi + dx/2

    # Set current (if it is on)
    Current = CurrentFunk(Xmean, Ymean)
    
    # Calculate time cost    
    Root = (1+Slope**2)*v**2 - Current**2
    if Root < 0: 
        # Set high cost for infeasible passage
        Cost = 1.0e6            
    else:
        # Calculate local cost for admissible passage
        Cost = dx*(1+Slope**2)/(np.sqrt(Root) - Slope*Current)
    return Cost

# Alternative version
def TimeCost_v2(xi,yi,xf,yf,v,CurrentFunk):
    D = (yf-yi)/(xf-xi)
    # The sign would have to be negative, it seems, to produce sensible results
    S = CurrentFunk((xi+xf)/2,(yi+yf)/2)
    Root = (1+D**2)*v**2-S**2
    if Root > 0:
        Time = (xf-xi)*(1+D**2)/(np.sqrt(Root) - D*S)
    else:
        Time = 1e-6    
    return Time

# Calculate total time for a given set of positions
def TotalTime(y,dx,dy,Ly,v,D,CurrentFunk):
    n = len(y)
    TimeHeur = 0
    # First step
    xi = 0
    yi = 0
    xf = dx
    yf = y[0]*dy-Ly/2
    TimeAddon = TimeCost_v2(xi,yi,xf,yf,v,CurrentFunk)
    TimeHeur += TimeAddon
    # Intermediate steps
    for i in range(1,n):
        xi = xf
        yi = yf
        xf = xi+dx
        yf = y[i]*dy-Ly/2
        TimeAddon = TimeCost_v2(xi,yi,xf,yf,v,CurrentFunk)
        TimeHeur += TimeAddon
    # Last step
    xi = xf
    yi = yf
    xf = D
    yf = 0
    TimeAddon = TimeCost_v2(xi,yi,xf,yf,v,CurrentFunk)
    TimeHeur += TimeAddon
    return TimeHeur

# Calculate crossing time from one point to next when there is no current
def TimeCostNoCurrent(xi,yi,Ystep,dx,dy,v):
    Slope = Ystep*dy/dx
    Root = (1+Slope**2)*v**2
    Cost = dx*(1+Slope**2)/(np.sqrt(Root))
    return Cost

#
# Various Hamiltonians (some are obsolete)
#

# Hamiltonian for boundary values
def Hbound(psi,d,n,dx,dy,v,kMid,TimeCost,CurrentFunk):
    """Implements boundary value on psi (left and right end)."""

    # Allocate
    Hpsi = np.zeros_like(psi)
    
    # Run over all elements
    for IndX in range(d**n):
        state = Index2State(IndX, d, n)
        # Update with boundary values
        YstepLeft = state[0] - kMid
        YstepRight = kMid - state[-1]
        # The corresponding costs 
        # (starting in (0,0) and ending in (D,0)).
        LeftCost = TimeCost(0,0,YstepLeft,dx,dy,v,CurrentFunk)
        RightCost = TimeCost(1-dx,0,YstepRight,dx,dy,v,CurrentFunk)
        Hpsi[IndX] = (LeftCost + RightCost)*psi[IndX]
        
    return Hpsi    


# Hamiltonian for hopping
def Hhop(psi, d, n):
    """Hopping between neighbouring internal states of each qudit."""

    # Allocate
    Hpsi = np.zeros_like(psi)

    # Loop over all states (paths)
    for IndX in range(d**n):

        # Convert to qudit string (basis state)
        state = Index2State(IndX, d, n)

        # Loop over all qudits
        for i in range(n):
            # k -> k+1, hop up
            if state[i] < d-1:
                NewState = state.copy()
                NewState[i] += 1
                NewInd = State2Index(NewState, d, n)
                Hpsi[NewInd] += psi[IndX]

            # k -> k-1, hop down
            if state[i] > 0:
                NewState = state.copy()
                NewState[i] -= 1
                NewInd = State2Index(NewState, d, n)
                Hpsi[NewInd] += psi[IndX]

    return Hpsi


# Implement the total cost as a diagonal operator - with current.
def Hcost(psi,d,n,dx,dy,Ly,v,kMid,CurrentFunk):
    """Implements the total cost for each path by summing
    the time penalty for each step in each path."""
    
    # Allocate
    Hpsi = np.zeros_like(psi)
    
    # Loop over all states (paths)
    for IndX in range(d**n):
        
        # Identify the corresponding qudit-string
        state = Index2State(IndX,d,n)

        # Update with boundary values
        YstepLeft = state[0] - kMid
        YstepRight = kMid - state[-1]
        yiRight = -Ly/2 + state[-1]*dy 
        # The corresponding costs 
        # (starting in (0,0) and ending in (D,0)).
        LeftCost = TimeCost(0,0,YstepLeft,dx,dy,v,CurrentFunk)
        RightCost = TimeCost(1-dx,yiRight,YstepRight,dx,dy,v,CurrentFunk)
        Hpsi[IndX] = (LeftCost + RightCost)*psi[IndX]

        # Add time-cost for each step, i.e., each 
        # neighbouring pair of qudits.
        for i in range(n-1):
            # (i,k) couples to all (i+1, k')            
            # Initial position
            xi = (i+1)*dx
            yi = -Ly/2 + state[i]*dy            
            # Calculate time cost
            Ystep = state[i+1]-state[i]
            Hpsi[IndX] += TimeCost(xi,yi,Ystep,dx,dy,v,CurrentFunk)*psi[IndX]

    return Hpsi



# Implement the total cost as a diagonal operator - NO current.
def HcostNoCurrent(psi,d,n,dx,dy,Ly,v,kMid):
    """Implements the total cost for each path by summing
    the time penalty for each step in each path."""
    
    # Allocate
    Hpsi = np.zeros_like(psi)
    
    # Loop over all states (paths)
    for IndX in range(d**n):
        
        # Identify the corresponding qudit-string
        state = Index2State(IndX,d,n)

        # Update with boundary values
        YstepLeft = state[0] - kMid
        YstepRight = kMid - state[-1]
        yiRight = -Ly/2 + state[-1]*dy 
        # The corresponding costs 
        # (starting in (0,0) and ending in (D,0)).
        LeftCost = TimeCostNoCurrent(0,0,YstepLeft,dx,dy,v)
        RightCost = TimeCostNoCurrent(1-dx,yiRight,YstepRight,dx,dy,v)
        Hpsi[IndX] = (LeftCost + RightCost)*psi[IndX]

        # Add time-cost for each step, i.e., each 
        # neighbouring pair of qudits.
        for i in range(n-1):
            # (i,k) couples to all (i+1, k')            
            # Initial position
            xi = (i+1)*dx
            yi = -Ly/2 + state[i]*dy            
            # Calculate time cost
            Ystep = state[i+1]-state[i]
            Hpsi[IndX] += TimeCostNoCurrent(xi,yi,Ystep,dx,dy,v) * psi[IndX]

    return Hpsi

# Hamiltonian for cost - non-diagonal old version
def HcostNonDiag(psi,d,n,dx,dy,Ly,v,kMid,CurrentFunk):
    """Sets a time penalty for each pair of neighbouring 
    qudits in a non-diagonal manner."""
    
    # Allocate
    Hpsi = np.zeros_like(psi)
    
    # Loop over all states (paths)
    for IndX in range(d**n):

        # Determine the corresponding qudit string (basis state)
        state = Index2State(IndX, d, n)

        # Update with boundary values
        YstepLeft = state[0] - kMid
        YstepRight = kMid - state[-1]
        yiRight = -Ly/2 + state[-1]*dy 
        # The corresponding costs 
        # (starting in (0,0) and ending in (D,0)).
        LeftCost = TimeCost(0,0,YstepLeft,dx,dy,v,CurrentFunk)
        RightCost = TimeCost(1-dx,yiRight,YstepRight,dx,dy,v,CurrentFunk)
        Hpsi[IndX] = (LeftCost + RightCost)*psi[IndX]

        # Forward coupling
        # qudit i couples to all possible states of qudit i+1
        for i in range(n-1):
            # (i,k) couples to all (i+1, k')            
            # Initial position
            xi = (i+1)*dx
            yi = -Ly/2 + state[i]*dy            
            for k in range(d):
                NewState = state.copy()
                NewState[i+1] = k
                NewInd = State2Index(NewState, d, n)
                # Calculate time cost
                Ystep = k-state[i]
                Hpsi[NewInd] += 0.5*TimeCost(xi,yi,Ystep,dx,dy,v,CurrentFunk) * \
                    psi[IndX]
                #print('Fram ',i,k,state,NewState, TimeCost(xi,yi,Ystep,dx,dy,
                #                                           v,CurrentFunk))    

        # Backward coupling
        # qudit i couples to all possible states of qudit i-1
        for i in range(1,n):
            # (i,k) couples to all (i-1, k')
            # Final position
            xf = (i+1)*dx
            xi = xf-dx
            for k in range(d):
                NewState = state.copy()
                NewState[i-1] = k
                NewInd = State2Index(NewState, d, n)
                # Calculate time cost
                Ystep = state[i]-k
                yi = -Ly/2 + k*dy
                Hpsi[NewInd] += 0.5*TimeCost(xi,yi,Ystep,dx,dy,v,CurrentFunk) * \
                    psi[IndX]
                #print('Bak ',i,k,state,NewState, TimeCost(xi,yi,Ystep,dx,dy,
                #                                          v,CurrentFunk))    

    return Hpsi


# New version of non-diagonal Hamiltonian
def HcostNonDiag_v2(psi,d,n,dx,dy,Ly,v,kMid,CurrentFunk):
    # Allocate
    Hpsi = np.zeros_like(psi)
    
    # Loop over all states (paths)
    for IndX in range(d**n):
        # Identify state
        State = Index2State(IndX,d,n)
        
        # Left boundary
        xi = 0
        xf = dx
        yi = 0
        yf = State[0]*dy - Ly/2
        TimePenalty = TimeCost_v2(xi,yi,xf,yf,v,CurrentFunk) 
        # Not sure about the factor 2; does it compensate adequately?
        Hpsi[IndX] += 1*TimePenalty*psi[IndX]
        
        # Right boundary
        xi = n*dx
        xf = xi+dx
        yi = State[-1]*dy - Ly/2
        yf = 0
        TimePenalty = TimeCost_v2(xi,yi,xf,yf,v,CurrentFunk) 
        # Not sure about the factor 2; does it compensate adequately?
        Hpsi[IndX] += 1*TimePenalty*psi[IndX]
        
        # Nearest neighbour couplings
        for i in range(1,n):
            for k in range(d):
                PrimeState = State.copy()
                PrimeState[i] = k
                PrimeIndX = State2Index(PrimeState,d,n)
                # Initial and final positions
                xi = i*dx
                yi = State[i-1]*dy - Ly/2
                xf = (i+1)*dx
                yf = k*dy - Ly/2
                TimePenalty = TimeCost_v2(xi,yi,xf,yf,v,CurrentFunk) 
                # Add coupling
                Hpsi[PrimeIndX] += TimePenalty * psi[IndX]
                Hpsi[IndX] += TimePenalty * psi[PrimeIndX]
        
    return Hpsi

#
# Hamiltonians corresponding to specific parts of the anneal sequence,
# uses the above ones
#

# First phase, in which there is no current and the hopping is turned on
def HamTurnOnHopping(psi,d,n,dx,dy,Ly,v,kMid,TimeCostNoCurrent,
                     AlphaMax,t,TimeDuration):
    return  t/TimeDuration*AlphaMax * Hhop(psi,d,n) + \
        HcostNoCurrent(psi,d,n,dx,dy,Ly,v,kMid,TimeCostNoCurrent)

# The current is turned on while the hopping is maximal
def HamTurnOnCurrent(psi,d,n,dx,dy,Ly,v,kMid,TimeCostNoCurrent,TimeCost,
                     CurrentFunk,AlphaMax,t,TimeDuration):
    return AlphaMax*Hhop(psi,d,n) +  \
        (1-t/TimeDuration) * HcostNoCurrent(psi,d,n,dx,dy,Ly,v,kMid, 
                                          TimeCostNoCurrent) + \
            t/TimeDuration * Hcost(psi,d,n,dx,dy,Ly,v,kMid,TimeCost, 
                                   CurrentFunk)
            
# The hopping is turned on while the current stays maximal
def HamTurnOffHopping(psi,d,n,dx,dy,Ly,v,kMid,TimeCost,
                     AlphaMax,CurrentFunk,t,TimeDuration):
    return  (1-t/TimeDuration)*AlphaMax * Hhop(psi,d,n) + \
        Hcost(psi,d,n,dx,dy,Ly,v,kMid,TimeCost,CurrentFunk)

