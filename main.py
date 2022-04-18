import math
import random
import numpy
import qConstants as qc
import qUtilities as qu
import qBitStrings as qb
import qGates as qg
import qMeasurement as qm
import qAlgorithms as qa

# It is conventional to have a main() function. Change it to do whatever you want. On Day 06 you could put your entanglement experiment in here.
def main():
    ketPsi = qu.uniform(2)
    print(ketPsi)

# If the user imports this file into another program as a module, then main() does not run. But if the user runs this file directly as a program, then main() does run.
if __name__ == "__main__":
    main()