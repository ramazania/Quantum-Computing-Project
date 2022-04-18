import math
import random
import numpy
import qConstants as qc
import qUtilities as qu
import qBitStrings as qb
import qGates as qg
import qMeasurement as qm
import qAlgorithms as qa


def bennett():
    '''Runs one iteration of the core algorithm of Bennett (1992). Returns a tuple of three items --- |alpha>, |beta>, |gamma> --- each of which is either |0> or |1>.'''
    #Alice and Bob randomly select a classical bit by flipping a coin.
    Alice = random.randint(0,1)
    Bob = random.randint(0,1)
    
    if Alice == 0:
        ketAlpha = qc.ket0
        ketPsi = qc.ket0
        if Bob == 0:
            ketBeta = qc.ket0
            measurement = qm.first(qg.tensor(ketPsi, qc.ket0))
            ketGamma = measurement[0]
        else:
            ketBeta = qc.ket1
            measurement = qm.first(qg.tensor(qg.application(qc.h, ketPsi), qc.ket0))
            ketGamma = measurement[0]
    else:
        ketAlpha = qc.ket1
        ketPsi = qc.ketPlus
        if Bob == 0:
            ketBeta = qc.ket0
            measurement = qm.first(qg.tensor(ketPsi, qc.ket0))
            ketGamma = measurement[0]
        else:
            ketBeta = qc.ket1
            measurement = qm.first(qg.tensor(qg.application(qc.h, ketPsi), qc.ket0))
            ketGamma = measurement[0]
    return (ketAlpha, ketBeta, ketGamma)
   

def deutsch(f):
    '''Implements the algorithm of Deutsch (1985). That is, given a two-qbit gate F representing a function f : {0, 1} -> {0, 1}, returns |1> if f is constant, and |0> if f is not constant.'''
    firstState = qg.tensor(qc.ket1, qc.ket1)
    hTensorh = qg.tensor(qc.h, qc.h)
    superposition = qg.application(hTensorh, firstState)
    applyF = qg.application(f, superposition)
    finalState = qg.application(hTensorh, applyF)
    measurement = qm.first(finalState)
    return measurement[0]


def bernsteinVazirani(n, f):
    '''Given n >= 1 and an (n + 1)-qbit gate F representing a function f : {0, 1}^n -> {0, 1} defined by mod-2 dot product with an unknown delta in {0, 1}^n, returns the list or tuple of n classical one-qbit states (each
    |0> or |1>) corresponding to delta.'''
    inputRegister = qg.power(qc.ket0, n)
    outputRegister = qc.ket1
    state = qg.tensor(inputRegister, outputRegister)
    hadamardLayer = qg.tensor(qg.power(qc.h, n), qc.h)
    superposition = qg.application(hadamardLayer, state)
    applyF = qg.application(f, superposition)
    finalState = qg.application(hadamardLayer, applyF)
    deltaList = []
    for i in range(n):
        measurement = qm.first(finalState)
        deltaList.append(measurement[0])
        finalState = measurement[1]
    return deltaList
    

def simon(n, f):
    '''The inputs are an integer n >= 2 and an (n + (n - 1))-qbit gate F
    representing a function f: {0, 1}^n -> {0, 1}^(n - 1) hiding an n-bit
    string delta as in the Simon (1994) problem. Returns a list or tuple of n
    classical one-qbit states (each |0> or |1>) corresponding to a uniformly
    random bit string gamma that is perpendicular to delta.'''
    inputRegister = qg.power(qc.ket0, n)
    outputRegister = qg.power(qc.ket0, n-1)
    hadamardLayer = qg.power(qc.h, n)
    superposition = qg.application(hadamardLayer, inputRegister)
    state = qg.tensor(superposition, outputRegister)
    #Applying the big F gate
    state = qg.application(f, state)
    #Measuring the output register after the big F gate is applied. 
    for i in range(n - 1):
        state = qm.last(state)[0]
    state = qg.application(hadamardLayer, state)
    #Measuring the input register
    deltaList = []
    for i in range(n):
        measurement = qm.first(state)
        deltaList.append(measurement[0])
        state = measurement[1]
    return deltaList
    

def shor(n, f):
    '''Assumes n >= 1. Given an (n + n)-qbit gate F representing a function
    f: {0, 1}^n -> {0, 1}^n of the form f(l) = k^l % m, returns a list or tuple
    of n classical one-qbit states (|0> or |1>) corresponding to the output of
    Shor’s quantum circuit.'''
    inputRegister = qg.power(qc.ket0, n)
    outputRegister = qg.power(qc.ket0, n)
    hadamardLayer = qg.power(qc.h, n)
    identityLayer = qg.power(qc.i, n)
    superposition = qg.application(hadamardLayer, inputRegister)
    outputRegister = qg.application(outputRegister, identityLayer)
    state = qg.tensor(superposition, outputRegister)
    #Applying the big F gate
    state = qg.application(f, state)
    #Measuring the output register after the big F gate is applied. 
    for i in range(n):
        state = qm.last(state)[0]
    state = qg.application(qg.fourier(n), state)
    #Measuring the input register
    stateList = []
    for i in range(n):
        measurement = qm.first(state)
        stateList.append(measurement[0])
        state = measurement[1]
    return stateList
    
    
def grover(n, k, f):
    '''Assumes n >= 1, k >= 1. Assumes that k is small compared to 2^n.
    Implements the Grover core subroutine. The F parameter is an (n + 1)-qbit
    gate representing a function f : {0, 1}^n -> {0, 1} such that
    SUM_alpha f(alpha) = k. Returns a list or tuple of n classical one-qbit
    states (either |0> or |1>), such that the corresponding n-bit string delta
    usually satisfies f(delta) = 1.'''  
    inputRegister = qg.power(qc.ket0, n)
    outputRegister = qc.ket1
    hadamardLayer = qg.power(qc.h, n+1)
    state = qg.tensor(inputRegister, outputRegister)
    superposition = qg.application(hadamardLayer, state)
    state = qg.application(f, superposition)  
    ketRho = qg.power(qc.ketPlus, n)
    identityLayer = qg.power(qc.i, n)
    R = 2 * numpy.outer(ketRho.conjugate(), ketRho) - identityLayer
    t = numpy.arcsin(math.sqrt(k) * 2**(n / -2))
    rotations = int(round(numpy.pi/(4*t) - 0.5))
    for i in range(rotations):
        state = qg.application(f, state)
        state = qg.application(qg.tensor(R, qc.i), state)
    deltaList = []
    for i in range(n):
        measurement = qm.first(state)
        deltaList.append(measurement[0])
        state = measurement[1]
    return deltaList
    
    
     

### DEFINING SOME TESTS ###

def bennettTest(m):
    # Runs Bennett's core algorithm m times.
    trueSucc = 0
    trueFail = 0
    falseSucc = 0
    falseFail = 0
    for i in range(m):
        result = bennett()
        if qu.equal(result[2], qc.ket1, 0.000001):
            if qu.equal(result[0], result[1], 0.000001):
                falseSucc += 1
            else:
                trueSucc += 1
        else:
            if qu.equal(result[0], result[1], 0.000001):
                trueFail += 1
            else:
                falseFail += 1
    print("check bennettTest for false success frequency exactly 0")
    print("    false success frequency = ", str(falseSucc / m))
    print("check bennettTest for true success frequency about 0.25")
    print("    true success frequency = ", str(trueSucc / m))
    print("check bennettTest for false failure frequency about 0.25")
    print("    false failure frequency = ", str(falseFail / m))
    print("check bennettTest for true failure frequency about 0.5")
    print("    true failure frequency = ", str(trueFail / m))

def deutschTest():
    def fNot(x):
        return (1 - x[0],)
    resultNot = deutsch(qg.function(1, 1, fNot))
    if qu.equal(resultNot, qc.ket0, 0.000001):
        print("passed deutschTest first part")
    else:
        print("failed deutschTest first part")
        print("    result = " + str(resultNot))
    def fId(x):
        return x
    resultId = deutsch(qg.function(1, 1, fId))
    if qu.equal(resultId, qc.ket0, 0.000001):
        print("passed deutschTest second part")
    else:
        print("failed deutschTest second part")
        print("    result = " + str(resultId))
    def fZero(x):
        return (0,)
    resultZero = deutsch(qg.function(1, 1, fZero))
    if qu.equal(resultZero, qc.ket1, 0.000001):
        print("passed deutschTest third part")
    else:
        print("failed deutschTest third part")
        print("    result = " + str(resultZero))
    def fOne(x):
        return (1,)
    resultOne = deutsch(qg.function(1, 1, fOne))
    if qu.equal(resultOne, qc.ket1, 0.000001):
        print("passed deutschTest fourth part")
    else:
        print("failed deutschTest fourth part")
        print("    result = " + str(resultOne))

def bernsteinVaziraniTest(n):
    delta = qb.string(n, random.randrange(0, 2**n))
    def f(s):
        return (qb.dot(s, delta),)
    gate = qg.function(n, 1, f)
    qbits = bernsteinVazirani(n, gate)
    bits = tuple(map(qu.bitValue, qbits))
    diff = qb.addition(delta, bits)
    if diff == n * (0,):
        print("passed bernsteinVaziraniTest")
    else:
        print("failed bernsteinVaziraniTest")
        print(" delta = " + str(delta))
        
def simonTest(n):
    # Pick a non-zero delta uniformly randomly.
    delta = qb.string(n, random.randrange(1, 2**n))
    # Build a certain matrix M.
    k = 0
    while delta[k] == 0:
        k += 1
    m = numpy.identity(n, dtype=int)
    m[:, k] = delta
    mInv = m
    # This f is a linear map with kernel {0, delta}. So it’s a valid example.
    def f(s):
        full = numpy.dot(mInv, s) % 2
        full = tuple([full[i] for i in range(len(full))])
        return full[:k] + full[k + 1:]
    gate = qg.function(n, n - 1, f)
    kets = simon(n, gate)
    bits = tuple(map(qu.bitValue, kets))
    if qb.dot(bits, delta) == 0:
    # if delta == prediction:
        print("passed simonTest")
    else:
        print("failed simonTest")
        # print(" delta = " + str(delta))
        # print(" prediction = " + str(prediction))

def shorTest(n, m):
    # Chooses a random k that is coprime to m
    k = 0
    2 <= k < m
    while math.gcd(k, m) != 1:
        k = random.randint(1, m)
        
    # Helper function: Builds the function f that computes powers of k modulo m
    def f(bit):
        integer = qb.integer(bit)
        powerK = qu.powerMod(k, integer, m)
        return qb.string(n, powerK)

    # Helper funtion: Converts a stateList into its corresponding stringList    
    def listToString(statelist):
        string = []
        for state in statelist:
            if state is qc.ket0:
                string.append(0)
            else:
                string.append(1)
        return string
    
    # Runs Shor’s quantum core subroutine on the corresponding gate F
    F = qg.function(n, n, f)
    result = shor(n, F)
    # Interprets the output as an integer b
    b = qb.integer(listToString(result))
    print(b)

    
def groverTest(n, k):
    # Pick k distinct deltas uniformly randomly.
    deltas = []
    while len(deltas) < k:
        delta = qb.string(n, random.randrange(0, 2**n))
        if not delta in deltas:
            deltas.append(delta)
    # Prepare the F gate.
    def f(alpha):
        for delta in deltas:
            if alpha == delta:
                return (1,)
        return (0,)
    fGate = qg.function(n, 1, f)
    # Run Grover’s algorithm up to 10 times.
    qbits = grover(n, k, fGate)
    bits = tuple(map(qu.bitValue, qbits))
    j = 1
    while (not bits in deltas) and (j < 10):
        qbits = grover(n, k, fGate)
        bits = tuple(map(qu.bitValue, qbits))
        j += 1
    if bits in deltas:
        print("passed groverTest in " + str(j) + " tries")
    else:
        print("failed groverTest")
        print(" exceeded 10 tries")
        print(" prediction = " + str(bits))
        print(" deltas = " + str(deltas))


# RUNNING THE TESTS ###

def main():
    bennettTest(100000)
    deutschTest()
    bernsteinVaziraniTest(4)
    simonTest(4)
    shorTest(5, 5)
    groverTest(5, 2)

    


if __name__ == "__main__":
    main()

