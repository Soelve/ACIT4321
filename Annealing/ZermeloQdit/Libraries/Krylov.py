"""These functions are related to Krylov-based methods for solving
the time-dependent Schrödinger equation."""

import numpy as np
from scipy.linalg import expm
from Libraries.Hamiltonians import HcostNoCurrent, Hcost, Hhop
from Libraries.Bookkeeping import StateNorm, InnerProduct   


# Propagator for the anneal. The phase, 1, 2 or 3, is an input.
def LanczosStepThreePhases(psi,t,dt,d,n,dx,dy,Ly,v,kMid,KrylovDim,
                    CurrentFunk,TimeHop,TimeCurrent,AlphaMax,phase):
    """This routine implements the Lanczos propagator and uses it to 
    propagate the state Psi a time-step dt."""

    # Allocation
    Vbig = np.zeros((d**n,KrylovDim+1), dtype=complex)
    Beta = np.zeros(KrylovDim+1, dtype=complex)
    Alfa = np.zeros(KrylovDim, dtype=complex)
    T = np.zeros((KrylovDim,KrylovDim))

    # Construct Lanczos basis
    Vbig[:,0]=psi                            # First state
    for j in range(KrylovDim):
        V = Vbig[:,j]
        # Apply Hamiltonian
        if phase ==1:
            U = HcostNoCurrent(V,d,n,dx,dy,Ly,v,kMid) + \
                t/TimeHop*AlphaMax*Hhop(V,d,n)
        elif phase ==2:
            U = (1-(t-TimeHop)/TimeCurrent)*HcostNoCurrent(V,d,n,
                                dx,dy,Ly,v,kMid) + \
                AlphaMax*Hhop(V,d,n) + \
                (t-TimeHop)/TimeCurrent*Hcost(V,d,n,dx,dy,Ly,v,kMid,
                                              CurrentFunk)    
        else:
            U = (1-(t-TimeHop-TimeCurrent)/TimeHop)* AlphaMax * \
                Hhop(V,d,n) + \
                Hcost(V,d,n,dx,dy,Ly,v,kMid,CurrentFunk)    
                
        # Remove component of V_{j-1}
        if j>0:
            U = U-Beta[j]*Vbig[:,j-1]
        
        Alfa[j] = InnerProduct(V,U,d,n)
        #Alfa[j] = np.sum(np.conj(V)*U)
        U = U-Alfa[j]*V
        Beta[j+1] = StateNorm(U,d,n)
        Vbig[:,j+1] = U/Beta[j+1]  

    # Build Lanczos representatoin of Hamiltonian
    T = np.diag(Alfa)
    T = T + np.diag(Beta[1:KrylovDim],1) + np.diag(Beta[1:KrylovDim],-1)

    # Exponentiate Hamiltonian to construct approximate propagator
    Ulanczos = expm(-1j*dt*T)
    # Apply to Psi - extract first column
    Ulanczos=Ulanczos[:,0]

    # Reconstruct propagated state
    PsiNew = np.zeros(d**n, dtype = complex)
    for k in range(KrylovDim):
        PsiNew += Ulanczos[k]*Vbig[:,k]
        
    return PsiNew


# Propagator for the Hcost term, imaginary time
def LanczosStepImTime(psi,dt,d,n,dx,dy,Ly,v,kMid,KrylovDim,CurrentFunk):
    """This routine implements the Lanczos propagator and uses it to 
    propagate the state Psi a time-step dt. This particular implementation 
    is used to test that it finds the ground state using imaginary time."""

    # Allocation
    Vbig = np.zeros((d**n,KrylovDim+1), dtype=complex)
    Beta = np.zeros(KrylovDim+1, dtype=complex)
    Alfa = np.zeros(KrylovDim, dtype=complex)
    T = np.zeros((KrylovDim,KrylovDim))

    # Construct Lanczos basis
    Vbig[:,0]=psi                            # First state
    for j in range(KrylovDim):
        V = Vbig[:,j]
        # Apply Hamiltonian
        U = Hcost(V,d,n,dx,dy,Ly,v,kMid,CurrentFunk)
        # Remove component of V_{j-1}
        if j>0:
            U = U-Beta[j]*Vbig[:,j-1]
        
        Alfa[j] = InnerProduct(V,U,d,n)
        #Alfa[j] = np.sum(np.conj(V)*U)
        U = U-Alfa[j]*V
        Beta[j+1] = StateNorm(U,d,n)
        #Beta[j+1] = np.sum(np.abs(U)**2)
        Vbig[:,j+1] = U/Beta[j+1]  

    # Build Lanczos representatoin of Hamiltonian
    T = np.diag(Alfa)
    T = T + np.diag(Beta[1:KrylovDim],1) + np.diag(Beta[1:KrylovDim],-1)

    # Exponentiate Hamiltonian to construct approximate propagator
    Ulanczos = expm(-dt*T)
    # Apply to Psi - extract first column
    Ulanczos=Ulanczos[:,0]

    # Reconstruct propagated state
    PsiNew = np.zeros(d**n, dtype = complex)
    for k in range(KrylovDim):
        PsiNew += Ulanczos[k]*Vbig[:,k]
        
    return PsiNew


