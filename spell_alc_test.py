import sys
import time
import argparse
from spell.fitting_alc import *

from spell.structures import solution2sparql, structure_from_owl
from spell.fitting import solve_incr, solve, mode



A1 = Structure(3,
      {
          "A" : {0,1},
          "B" : {0,2}
      },
      {i : {} for i in range(3)},{},{}
)
P1 = [0]
N1 = [1,2]

A3 = (3,
      {
          "A" : {1},
          "B" : {2}
      },
      {i : {} for i in range(3)}
)
P3 = [1,2]
N3 = [0]


rn2= {i : {} for i in range(3)}
rn2[0] = {(2,'r')}
cn2 = {
          "A" : {0,1,2},
          "B" : {0,1}
      }
A2 = Structure(3, cn2,rn2, {},{})


P2 = [0]
N2 = [1]


def main():
    i1 = (A1,3,P1,N1)   
    i2 = (A2,3,P2,N2)      
    f = FittingALC(*i2, op = {AND,OR,EX,ALL})
    #f = FittingALC(*i1, op = {AND,OR,EX})
    if(f.solve()):
        print(f._modelToTree())

if __name__ == "__main__":
    main()