import random
import numpy as np
import scipy.stats

r = random.Random()
l = (100, 1000, 10000)
samples = [[r.gauss(0,1) for i in range(l[j])] for j in range(3)]
errs = (0.05, 0.01, 0.001)
qnorm05 = scipy.stats.norm.ppf(1-errs[0]/2)
qnorm01 = scipy.stats.norm.ppf(1-errs[1]/2)
qnorm001 = scipy.stats.norm.ppf(1-errs[2]/2)
qt05 = [scipy.stats.t.ppf(1-errs[0]/2, l[i]-1) for i in range(3)]
qt01 = [scipy.stats.t.ppf(1-errs[1]/2, l[i]-1) for i in range(3)]
qt001 = [scipy.stats.t.ppf(1-errs[2]/2, l[i]-1) for i in range(3)]
qchi05 = [(scipy.stats.chi2.ppf(1-errs[0]/2, l[i]-1), scipy.stats.chi2.ppf(errs[0]/2, l[i]-1)) for i in range(3)]
qchi01 = [(scipy.stats.chi2.ppf(1-errs[1]/2, l[i]-1), scipy.stats.chi2.ppf(errs[1]/2, l[i]-1)) for i in range(3)]
qchi001 = [(scipy.stats.chi2.ppf(1-errs[2]/2, l[i]-1), scipy.stats.chi2.ppf(errs[2]/2, l[i]-1)) for i in range(3)]
print "chi2 quantiles:"
print qchi05
# clt
# P( -z < (sum(X)-na)/(var(x)*n)**0.5 < z) = 1 - 2(1-F(z)) = 0.95
# 0.05 = 2 - 2F(z)
# 0.025 = 1 - F(z)
# z = q(0.975)
# P( -z*(var(x)*n)**0.5 + sum(X) < na < z*(var(x)*n)**0.5 + sum(X)) = 0.95

def task1():
    means = [np.mean(samples[i]) for i in range(3)]
    S = [(np.var(samples[i])*l[i]/(l[i]-1))**0.5 for i in range(3)]
    i05 = [((-qt05[i]*S[i])/(l[i])**0.5 + means[i], (qt05[i]*S[i])/(l[i])**0.5 + means[i]) for i in range(3)]
    i01 = [((-qt01[i]*S[i])/(l[i])**0.5 + means[i], (qt01[i]*S[i])/(l[i])**0.5 + means[i]) for i in range(3)]
    i001 = [((-qt001[i]*S[i])/(l[i])**0.5 + means[i], (qt001[i]*S[i])/(l[i])**0.5 + means[i]) for i in range(3)]
    print "Estimation mean confidency interval via t-distribution:"
    print "confidency 0.95: ", i05
    print "confidency 0.99: ", i01
    print "confidency 0.999: ", i001
    print "---"

def task2():
    means = [np.mean(samples[i]) for i in range(3)]
    var = [np.var(samples[i]) for i in range(3)]
    i05 = [(-qnorm05*(var[i]/l[i])**0.5 + means[i], qnorm05*(var[i]/l[i])**0.5 + means[i]) for i in range(3)]
    i01 = [(-qnorm01*(var[i]/l[i])**0.5 + means[i], qnorm01*(var[i]/l[i])**0.5 + means[i]) for i in range(3)]
    i001 = [(-qnorm001*(var[i]/l[i])**0.5 + means[i], qnorm001*(var[i]/l[i])**0.5 + means[i]) for i in range(3)]
    print "Estimation mean confidency interval via CLT:"
    print "confidency 0.95: ", i05
    print "confidency 0.99: ", i01
    print "confidency 0.999: ", i001
    print "---"

def task3():
    nvar = [np.var(samples[i])*l[i] for i in range(3)]
    i05 = [(nvar[i]/qchi05[i][0], nvar[i]/qchi05[i][1]) for i in range(3)]
    i01 = [(nvar[i]/qchi01[i][0], nvar[i]/qchi01[i][1]) for i in range(3)]
    i001 = [(nvar[i]/qchi001[i][0], nvar[i]/qchi001[i][1]) for i in range(3)]
    print "Estimation variance confidency interval via chi-square:"
    print "confidency 0.95: ", i05
    print "confidency 0.99: ", i01
    print "confidency 0.999: ", i001
    print "---"

if __name__ == '__main__':
    task1()
    task2()
    task3()
