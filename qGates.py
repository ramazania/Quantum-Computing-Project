import math
import random
import numpy
import qConstants as qc
import qUtilities as qu
import qBitStrings as qb
import qGates as qg
import qMeasurement as qm
import qAlgorithms as qa


def application(u, ketPsi):
    '''Assumes n >= 1. Applies the n-qbit gate U to the n-qbit state |psi>, returning the n-qbit state U |psi>.'''
    return numpy.dot(u, ketPsi)


def tensor(a, b):
   return numpy.kron(a,b)


def function(n, m, f):
    '''Assumes n, m == 1. Given a Python function f : {0, 1}^n -> {0, 1}^m.
That is, f takes as input an n-bit string and produces as output an m-bit
string, as defined in qBitStrings.py. Returns the corresponding
(n + m)-qbit gate F.'''
    alpha = n * (0,)
    beta = m * (0,)
    F = numpy.zeros((2**(n+m), 2**(n+m)), dtype=numpy.array(0 + 0j).dtype)
    #Helper function to convert bits to their corresponding quantum states
    def g(bit):
        if bit == 0:
            return qc.ket0
        else:
            return qc.ket1
    #Helper function to make state from string. 
    def stringToState(alpha):
        state = g(alpha[0])
        for i in range(1, len(alpha)):
            state = tensor(state, g(alpha[i]))
        return state
    #Looping over alpha and beta to make the big gate F.
    for i in range(2**n):
        ketAlpha = stringToState(alpha)
        for j in range (2**m):
            ketsum = stringToState(qb.addition(beta, f(alpha)))
            ketGamma = tensor(ketAlpha, ketsum)
            #Coverting bitstring to its corresponding integer. 
            integer = qb.integer(alpha + beta)
            #Making the columns of big F gate.
            F[integer] = ketGamma
            beta = qb.next(beta)
        alpha = qb.next(alpha)
    return F


def power(stateOrGate, m):
    '''Assumes n >= 1. Given an n-qbit gate or state and m >= 1, returns the
    mth tensor power, which is an (n * m)-qbit gate or state. For the sake of
    time and memory, m should be small.'''
    if m == 1:
        return stateOrGate
    else:
        temp = stateOrGate
        for i in range(m-1):
            temp = tensor(temp, stateOrGate)
        return temp


def fourier(n):
    '''Assumes n >= 1. Returns the n-qbit quantum Fourier transform gate T.'''
    T = numpy.zeros((2**(n), 2**(n)), dtype=numpy.array(0 + 0j).dtype)
    normalizationConstant = (1 / 2**(n / 2))
    for alpha in range(2**n):
        for beta in range(2**n):
            T[alpha, beta] = numpy.exp(1j * 2 * numpy.pi * alpha * beta / 2**n)      
    return  normalizationConstant * T

    
    
    
    
    
    
### DEFINING SOME TESTS ###

def applicationTest():
    # These simple tests detect type errors but not much else.
    answer = application(qc.h, qc.ketMinus)
    if qu.equal(answer, qc.ket1, 0.000001):
        print("passed applicationTest first part")
    else:
        print("FAILED applicationTest first part")
        print("    H |-> = " + str(answer))
    ketPsi = qu.uniform(2)
    answer = application(qc.swap, application(qc.swap, ketPsi))
    if qu.equal(answer, ketPsi, 0.000001):
        print("passed applicationTest second part")
    else:
        print("FAILED applicationTest second part")
        print("    |psi> = " + str(ketPsi))
        print("    answer = " + str(answer))


def tensorTest():
    # Pick two gates and two states.
    u = qc.x
    v = qc.h
    ketChi = qu.uniform(1)
    ketOmega = qu.uniform(1)
    # Compute (U tensor V) (|chi> tensor |omega>) in two ways.
    a = tensor(application(u, ketChi), application(v, ketOmega))
    b = application(tensor(u, v), tensor(ketChi, ketOmega))
    # Compare.
    if qu.equal(a, b, 0.000001):
        print("passed tensorTest")
    else:
        print("FAILED tensorTest")
        print("    a = " + str(a))
        print("    b = " + str(b))


def functionTest(n, m):
    # 2^n times, randomly pick an m-bit string.
    values = [qb.string(m, random.randrange(0, 2**m)) for k in range(2**n)]
    # Define f by using those values as a look-up table.
    def f(alpha):
        a = qb.integer(alpha)
        return values[a]
    # Build the corresponding gate F.
    ff = function(n, m, f)
    # Helper functions --- necessary because of poor planning.
    def g(gamma):
        if gamma == 0:
            return qc.ket0
        else:
            return qc.ket1
    def ketFromBitString(alpha):
        ket = g(alpha[0])
        for gamma in alpha[1:]:
            ket = tensor(ket, g(gamma))
        return ket
    # Check 2^n - 1 values somewhat randomly.
    alphaStart = qb.string(n, random.randrange(0, 2**n))
    alpha = qb.next(alphaStart)
    while alpha != alphaStart:
        # Pick a single random beta to test against this alpha.
        beta = qb.string(m, random.randrange(0, 2**m))
        # Compute |alpha> tensor |beta + f(alpha)>.
        ketCorrect = ketFromBitString(alpha + qb.addition(beta, f(alpha)))
        # Compute F * (|alpha> tensor |beta>).
        ketAlpha = ketFromBitString(alpha)
        ketBeta = ketFromBitString(beta)
        ketAlleged = application(ff, tensor(ketAlpha, ketBeta))
        # Compare.
        if not qu.equal(ketCorrect, ketAlleged, 0.000001):
            print("failed functionTest")
            print(" alpha = " + str(alpha))
            print(" beta = " + str(beta))
            print(" ketCorrect = " + str(ketCorrect))
            print(" ketAlleged = " + str(ketAlleged))
            print(" and here’s F...")
            print(ff)
            return
        else:
            alpha = qb.next(alpha)
    print("passed functionTest")


def fourierTest(n):
    if n == 1:
        # Explicitly check the answer.
        t = fourier(1)
        if qu.equal(t, qc.h, 0.000001):
            print("passed fourierTest")
        else:
            print("failed fourierTest")
            print(" got T = ...")
            print(t)
    else:
        t = fourier(n)
        # Check the first row and column.
        const = pow(2, -n / 2) + 0j
        for j in range(2**n):
            if not qu.equal(t[0, j], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        for i in range(2**n):
            if not qu.equal(t[i, 0], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        print("passed fourierTest first part")
    # Check that T is unitary.
    tStar = numpy.conj(numpy.transpose(t))
    tStarT = numpy.matmul(tStar, t)
    id = numpy.identity(2**n, dtype=qc.one.dtype)
    if qu.equal(tStarT, id, 0.000001):
        print("passed fourierTest second part")
    else:
        print("failed fourierTest second part")
        print(" T^* T = ...")
        print(tStarT)

### RUNNING THE TESTS ###

def main():
    applicationTest()
    applicationTest()
    tensorTest()
    tensorTest()
    functionTest(3, 3)
    fourierTest(4)


if __name__ == "__main__":
    main()
