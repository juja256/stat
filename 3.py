# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

class MarkovChain:
    def __init__(self, P, start):
        self.P = P
        self.start = start
        self.dim = len(start)
        self.E = range(self.dim)
        self.cur = np.random.choice(self.E, 1, True, self.start)[0]
        
        
    def next(self):
        cur_P = self.P[self.cur]
        self.cur = np.random.choice(self.E, 1, True, cur_P)[0]
        return self.cur
    
    def touch(self):
        return self.cur
        
        
        
def task1():
    gamma = 0.05
    N = 1000
    p = 2
    M = N*p
    X = np.sort(np.random.exponential(2, N))
    Y = np.random.exponential(2, M)
    q_norm = st.norm.ppf(1-gamma/2)
    nu = np.array([ np.sum(Y <= X[i+1]) - np.sum( Y < X[i]) for i in range(0, N-1)])
    nu = np.concatenate( ( [ np.sum(Y <= X[0]) ], nu, [ np.sum(Y > X[-1] ) ] ) )
    mu = np.count_nonzero(nu == 0)
    a = N/(1+p)
    D = (N*p**2) / (1+p)**3
    mu_norm = (mu - a)/((D)**0.5)
    
    print("Testing homogeneousness via empty blocks test")
    print(mu, mu_norm, q_norm)
    if abs(mu_norm) < q_norm:
        print("X and Y are sampled from the same distribution")
    else:
        print("X and Y are not sampled from the same distribution")


def task2():
    gamma = 0.05
    N = [2000, 3000, 8000]
    k=len(N)
    X = np.array([np.random.poisson(3, n) for n in N])
    #print(np.concatenate(X))
    a = np.mean(np.concatenate(X))
    r = 10
    p_main = st.poisson(a).pmf(range(r-1))
    p = np.append(p_main , 1-np.sum(p_main))
    nu = np.zeros(shape=(k,r))
    
    for i in range(k):
        for j in range(r):
            a = (X[i] >= j) if j == r-1 else (X[i] == j)
            nu[i,j] = np.count_nonzero(a)
        
    m = np.matrix(N).T * np.matrix(p)
    chi2_stat = np.sum((nu-m.A)**2 / m)
    q_chi2 = st.chi2((r-1)*k-1).ppf(1-gamma)
    print("Testing homogeneousness via chi2 test")
    print(nu)
    print(chi2_stat, q_chi2)
    if chi2_stat < q_chi2:
        print("X[0], X[1], X[2] are sampled from the same distribution")
    else:
        print("X[0], X[1], X[2] are not sampled from the same distribution")
            
            
def task3():
    N = 10000
    gamma = 0.01
    
    X = np.random.poisson(2, N)
    Z = np.random.poisson(2, N)
    Y = np.random.poisson(5, N)
    #Y = X+Z
    r = 8
    k = 15
    q_chi2 = st.chi2((r-1)*(k-1)).ppf(1-gamma)
    nu = np.zeros(shape=(r,k))
    for i in range(r):
        for j in range(k):
            a = (X >= i) if i == r-1 else (X == i)
            b = (Y >= j) if j == k-1 else (Y == j)
            nu[i, j] = np.count_nonzero(a & b)
    
    nu_r = [np.sum(nu[i,:]) for i in range(r)]
    nu_k = [np.sum(nu[:,j]) for j in range(k)]
    print("Testing independence via chi2 test")
    print(nu_k, nu_r)
    print(nu)
    chi2=0
    for i in range(r):
        for j in range(k):
            p = nu_r[i]*nu_k[j]/N
            if (p):
                chi2 += (nu[i,j] - p)**2 / p
    
    print(chi2, q_chi2)
    
    if chi2 < q_chi2:
        print("X and Y are independent")
    else:
        print("X and Y are not independent")
    #chi2_stat = np.sum()


def task4():
    N = 1000
    gamma = 0.05
    q_norm = st.norm.ppf(1-gamma/2)
    
    X = np.random.logistic(0, 1, N)
    Z = np.random.uniform(-1, 1, N)
    Y = np.random.normal(0, 1, N)
    #Y = X+Z
    R = st.rankdata(X, method='ordinal')
    S = st.rankdata(Y, method='ordinal')
    plt.plot(S, R, '.')
    #plt.plot(range(N), S, '.')
    plt.show()
    plt.clf()
    print("Testing independence via Spearman's test")
    print(R)
    print(S)
    rho = 1 - (6/(N*(N**2-1))) * np.sum((R - S)**2)
    rho_norm = rho * N**0.5
    print(rho, rho_norm, q_norm)
    if abs(rho_norm) < q_norm:
        print("X and Y are independent")
    else:
        print("X and Y are not independent")


def task5():
    N = 1000
    gamma = 0.05
    q_norm = st.norm.ppf(1-gamma/2)
    X = np.random.uniform(0, 1, N)
    seed = int(29*X[0])
    for i in range(1, N):
        seed = 2 ** seed % 29
        X[i] = seed/29
    
    #print(X)
    eta = np.zeros(N-1)
    for i in range(N-1):
        for j in range(i+1, N-1):
            eta[i] += 1 if X[i] > X[j] else 0
    plt.plot(range(N-1), eta, '.')
    plt.show()
    plt.clf()
    S = np.sum(eta)
    S_norm = (S - N*(N-1)/4) #*(6/N**1.5)
    print("Testing randomness via inversion test")
    print(S, S_norm, S_norm*6/(N**1.5), q_norm)
    if abs(S_norm) < q_norm * (N**1.5) / 6:
        print("Sample is statistical random")
    else:
        print("Sample is not statistical random")


def task6():
    def index_of(arr):
        idx = 0
        for k in range(len(arr)):
            idx += int(arr[k] * 10**k)
        return idx
    
    S = 4
    M = 10000
    gamma = 0.05
    start = np.array([0,0,1,0])
    P = np.array([[0.5, 0.5, 0, 0],
                  [0, 0, 0.6, 0.4],
                  [0.2, 0.2, 0.3, 0.3],
                  [0.5, 0.3, 0.1, 0.1]])
    mc = MarkovChain(P, start)
    X = np.trunc( 10 * np.random.uniform(0,1,M) )
    #X = np.array( [mc.next() for i in range(M)] )
    q_chi2 = st.chi2( 9 * 10**S).ppf(1-gamma)
    freq_table = np.zeros(10**(S+1))
    freq_marginal = np.zeros(10**S)
    for k in range(M-S):
        freq_table[index_of( X[k:k+S+1] )] += 1
        freq_marginal[index_of(X[k:k+S])] += 1
    freq_marginal[index_of(X[-S:])] += 1
    plt.plot(range(10**S), freq_marginal, '.')
    plt.show()
    plt.clf()
    chi2 = 0
    for k in range(10):
        for i in range(10**S):
            if freq_marginal[i]:
                chi2 += (freq_table[i + k * 10**S] - freq_marginal[i]/10)**2/(freq_marginal[i]/10)
    
    print("Testing non-random S-chaining via chi2 test")
    print(chi2, q_chi2)
    if chi2 > q_chi2:
        print("Sequence forms a S-Markov chain")
    else:
        print("Sequence does not form a S-Markov chain")
    
    
def main():
    task1()
    task2()
    task3()
    task4()
    task5()
    task6()
    
if __name__ == '__main__':
    main()