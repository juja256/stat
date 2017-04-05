# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# np > 5
# p > 5/n

def task3():
    gamma = 0.05
    N = 1000
    M = 1000
    ks = st.kstwobign.ppf(1-gamma)
    X = np.sort(np.random.normal(0.0, 1, N))
    Y = np.sort(np.random.normal(0.0, 1, M))
    EDF_X = lambda u: (1/N)*np.sum(X < u)
    
    D1 = np.max([ (k/M) - EDF_X(Y[k-1]) for k in range(1, M+1) ])
    D2 = np.max([ EDF_X(Y[k-1]) - (k-1)/M for k in range(1, M+1) ])
    D = ((N*M/(N+M))**0.5) * np.max([D1, D2])
    print("Testing distribution equality via K-S test:")
    print(D, ks)
    if (D < ks):
        print("Samples are distributed identically")
    else:
        print("Samples are not distributed identically")


def task2():
    gamma = 0.05
    N = 1000
    p = 2
    q_norm = st.norm.ppf(1-gamma)
    r = N//p
    U = np.arange(0, 1+1/r, 1/r)
    X = np.random.normal(0, 1, N)
    X_un = st.norm(0,1).cdf(X)
    nu = np.array([ np.sum(X_un < U[i+1]) - np.sum( X_un < U[i]) for i in range(0, r)])
    mu = np.count_nonzero(nu == 0)  
    
    mu_norm = (mu - r*np.exp(-p))/((r*( np.exp(-p)*(1-np.exp(-p)) - p*np.exp(-2*p) ))**0.5)
    print("Testing distribution function for sample via empty cells test")
    print(mu, mu_norm, q_norm)
    if mu_norm < q_norm:
        print("Sample are from distribution F")
    else:
        print("Sample are not from distribution F")


def task1():
    N = 10000
    gamma = 0.05
    X = np.random.normal(0.0, 1, N)
    U = np.array([])
    u_low = st.norm(0,1).ppf(16./N)
    u_high = -u_low
    r = 30
    dx = 2*u_high/(r-1)
    U = np.arange(u_low, u_high+dx, dx)
    print(U)
    pn = (st.norm(0,1).cdf(U[1:]) - st.norm(0,1).cdf(U[:-1])) *N
    pn = np.concatenate( ( [st.norm(0,1).cdf(U[0])*N], pn, [st.norm(0,1).cdf(U[0])*N] ) )
    print(pn)
    
    nu = np.concatenate( ([ np.sum(X < U[0]) ] ,
                           [ np.sum(X < U[i+1]) - np.sum( X < U[i]) for i in range(0, r-1)]
                           , [ np.sum( X > U[-1]) ]) )
    
    print(nu)
    plt.plot(np.concatenate( ( U ,[U[-1]+dx] ) ), nu, '.-')
    plt.show()
    chi2 = np.sum( (nu - pn)**2 / pn )
    chi2_crit = st.chi2(r).ppf(1-gamma)
    print("Checking sample distribution N(0,1) via chi2:")
    print("Chi2 statistic and critical:", chi2, chi2_crit)
    if chi2 < chi2_crit:
        print("Sample is from N(0,1)")
    else:
        print("Sample is not from N(0,1)")


def main():
    task1()
    task2()
    task3()
        
    
if __name__ == '__main__':
    main()