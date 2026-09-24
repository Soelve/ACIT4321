"""This is a library of functions used to do the bookeeping. It converts
between qudit-string-notation and global index, it calculates the norm,
it gathers statistics for the various qudits ..."""

import numpy as np

# Convert from qutrit string to global index
def State2Index(State, d, n):
    """Convert state-array [k_1, ..., k_n] inot running index I.
    Inputs:
    I - the index, runs from 0 to d^n-1
    d - the dimension of the single qudit space; d=2 is a qubit
    n - the number of qudits."""
    
    if (len(State) != n):
        raise ValueError("Inconsistent dimensions")
    
    I  = 0
    power = 0
    for k in range(n):
        I += State[n-k-1]*d**power
        power += 1
    return int(I)

# Convert from global index to qutrit string, i.e., basis state or path 
def Index2State(Ind, d, n):
    """Convert running index I to state-array [k_1, ..., k_n].
    Inputs:
    I - the index, runs from 0 to d^n-1
    d - the dimension of the single qudit space; d=2 is a qubit
    n - the number of qudits.
    """
       
    if (Ind > d**n-1):
        raise ValueError("Too large an index")
    
    State = np.zeros(n, dtype=int)
    
    for k in range(n):
        q = Ind // d**(n-k-1)
        State[k] = q
        Ind -= q*d**(n-k-1)

    return State

# Function for calculating the norm of our state
def StateNorm(psi, d, n):
    
    ProbSum = 0
    
    # Loop over all states
    for i in range(d**n):
        ProbSum += np.abs(psi[i])**2
            
    return np.sqrt(ProbSum)

# Function for calculating the inner product between two states
def InnerProduct(psi1, psi2, d, n):
    
    InnerProd = 0
    
    # Loop over all states
    for i in range(d**n):
        InnerProd += np.conj(psi1[i]) * psi2[i]
            
    return InnerProd


# Function for extracting statistic for individual qutrits.
def QuditDist(psi, d, n):
    # Allocate
    Distribution = np.zeros((d,n))
    
    # Loop over all states
    for i in range(d**n):
        # Find probability addition
        Probability = np.abs(psi[i])**2
        # Identify corresponding basis state
        state = Index2State(i, d, n)
        # Update the entries in question
        for j in range(n):
            Distribution[state[j],j] += Probability
            
    return Distribution