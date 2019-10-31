import numpy as np
import matplotlib.pyplot as plt

def interp(ni, nj, N):
    '''
    Returns an interpolation function from bin ni to bin nj (been N the total number of bins) in polar coordinates
    See https://calculus7.org/2013/04/20/peanut-allergy-and-polar-interpolation/
    '''
    n1 = min(ni, nj)
    n2 = max(ni, nj)
    Nm = int(N/2-0.0001)
    if (n2-n1 < N/2):
        r = ((Nm+1-n2+n1)/(Nm+1))**(-2)
        delta = np.pi/N*(n2-n1)
        theta0 = 2*n1*np.pi/N+delta
    elif (n2-n1 > N/2):
        r = ((Nm+1+n2-n1-N)/(Nm+1))**(-2)
        delta = np.pi/N*(N-n2+n1)
        theta0 = 2*n1*np.pi/N - delta
    else:
        r = 1e10 #yeah, I don't know why this works :O
        delta = np.pi/N*(n2-n1)
        theta0 = 2*n1*np.pi/N+delta

    return lambda x: (r + (1-r)*np.sin(x-theta0)**2/np.sin(delta)**2)**(-0.5)


def cov2corr(cov):
    '''
    Computes the correltion matrix from the covariance matrix
    '''
    return np.diag((np.diag(cov))**(-0.5)) @ cov @ np.diag((np.diag(cov))**(-0.5))

def circlecorr(cov, R=10, maxlw=5, labels=None):
    '''
    Circular plot for the correlation matrix.
    The bins represent the variance of each variable,
    and the lines the correlation coefficient:
    the width of the line corresponds to the absolute value of the correlation coefficient.
    Blue lines indicate negative correlation,
    Red lines indicate positive correlation.

    Function arguments:
    - cov: Covariance matrix.
    - R: Radius of the circle that contains the base of the bins.
    - maxlw: Linewidth of the lines of correlation = +-1
    - labels: list of labels for each bin
    '''
    N = cov.shape[0]
    corr = cov2corr(cov)

    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    width = 2*np.pi/(N*1.2)
    ax = plt.subplot(111, polar= True)
    ax.set_xticks(np.linspace(0, 2*np.pi, 6, endpoint=False))
    if labels is not None:
        ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    bars = ax.bar(theta, np.diag(cov), width=width, bottom=R)
    for i in range(N):
        for j in range(i+1, N):
            if (j-i <= N/2):
                lin =np.linspace( 2*i*np.pi/N, 2*j*np.pi/N, 100 )
            else:
                lin =np.linspace( 2*j*np.pi/N, 2*(i+N)*np.pi/N, 100 )
            int1 = interp(i,j,N)
            if corr[i,j] > 0:
                c = 'red'
            else:
                c = 'blue'
            plt.plot(lin, R*int1(lin), c=c, lw=maxlw*abs(corr[i,j]))
