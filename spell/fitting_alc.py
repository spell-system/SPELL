import time
from enum import Enum
from typing import NamedTuple, Union
from asciitree import LeftAligned
from asciitree.drawing import BoxStyle, Style
from collections import OrderedDict as OD


from pysat.card import CardEnc, EncType
from pysat.solvers import Glucose4, pysolvers

from .structures import (
    Signature,
    Structure,
    conceptname_ext,
    conceptnames,
    generate_all_trees,
    ind,
    restrict_to_neighborhood,
    rolenames,
    solution2sparql,
)

from .fitting import (
    determine_relevant_symbols
)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

d_op = {
     0 : "TOP",
     1 : "BOT",
     2: "NEG",
     3: "AND",
     4: "OR",
     5: "EX",
     6: "ALL"
}
d_var_names = {
    0:"X",
    1:"Y",
    2:"Z",
    3:"U",
    4:"V",    
}
TOP = 0
BOT = 1
NEG = 2
AND = 3
OR = 4
EX = 5
ALL = 6
ALC_OP = {NEG,AND,OR,EX,ALL}
ALC_OP_B = {NEG,AND,OR}
X = 0
Y = 1
Z = 2
U = 3
V = 4

class STreeNode():
    def __init__(self, node, children):
        self.node = node
        self.children = children
    
    def _to_OD_c(self):
            if not self.children:
                return OD()
            else:
                return OD(list(map(lambda x : (x.node,x._to_OD_c()),self.children)))
            
    def _to_OD(self):                
        return OD([(self.node,self._to_OD_c())])

    def to_asciitree(self):
        tr = LeftAligned()
        tr.draw = BoxStyle(node_label = lambda x : x[1])
        return tr(self._to_OD())
    
    @classmethod
    def FromDict(cls,dict,root):
        return cls(root, list(map(lambda x : cls.FromDict(dict, x), dict[root])))

class FittingALC:
    def __init__(self, A: Structure, k : int, P: list[int],
        N: list[int], op = ALC_OP, cov_p = - 1, cov_n = -1):
        B,m = restrict_to_neighborhood(k-1,A,P + N)
        self.P = [m[a] for a in P]
        self.N = [m[b] for b in N]
        self.A = B
        self.sigma = determine_relevant_symbols(A, P, 1, k - 1, N)
        self.k = k
        self.op = op
        self.op_b = ALC_OP_B.intersection(op)
        self.op_r = op.difference(ALC_OP_B)
        self.tree_node_symbols = dict()
        self.vars = self._vars()
        self.n_op = len(op)
        self.cov_p = len(P) if cov_p == -1 else cov_p
        self.cov_n = len(N) if cov_n == -1 else cov_n
        self.solver = Glucose4()
        

    def _vars(self):
        d = dict()
        i = 1
        d[X,TOP] = i
        self.tree_node_symbols[i] = d_op[TOP]
        d[X,BOT] = i * self.k + 1
        self.tree_node_symbols[i * self.k+1] = d_op[BOT]
        i+=1
        for cn in self.sigma[0]:
            d[X,cn] = i * self.k+1
            self.tree_node_symbols[i * self.k+1] = cn
            i += 1
        for op in self.op_b:
            d[X,op]=i * self.k+1
            self.tree_node_symbols[i * self.k+1] = d_op[op]
            i+=1
        if EX in self.op:
            for c in self.sigma[1]:
                d[X,EX,c] = i * self.k+1
                self.tree_node_symbols[i * self.k+1] = f"ex.{c}"
                i += 1
        if ALL in self.op:
            for c in self.sigma[1]:
                d[X,ALL,c] = i * self.k+1
                self.tree_node_symbols[i * self.k+1] = f"all.{c}"
                i+=1
        for l in range(self.k):
            d[Y,l] = i * self.k+1
            i +=1
        for a in range(self.A.max_ind):
            d[Z,a] = i * self.k+1
            i += 1
        for op in self.op_b:
            for j in range(self.k):
                for a in range(self.A.max_ind):
                    d[U,op,j,a] = i * self.k+1
                    i+=1
        for op in self.op_r:
            for j in range(self.k):
                for a in range(self.A.max_ind):
                    d[U,op,j,a] = i*self.k+1
                    i+=1
        for j in range(self.k):
            d[V,1,j] = i*self.k+1
            i+=1
        for j in range(self.k):
            d[V,2,j] = i*self.k+1
            i+=1
        return d

    def _root(self):
        for j in range(1,self.k):
            self.solver.add_clause([-self.vars[Y,j]])

    def _syn_tree_encoding(self):
        for i in range(self.k):            
            x_vars = [self.vars[X,o]+i for o in self.op_b] + [self.vars[X,o,r]+i for o in self.op_r for r in self.sigma[1]] + [self.vars[X,cn] +i for cn in self.sigma[0]] + [self.vars[X,TOP]+i,self.vars[X,BOT]+i]
            self.solver.add_clause(x_vars)
            for v1 in x_vars:
                for v2 in x_vars:
                    if v1 != v2:
                        self.solver.add_clause((-v1,-v2))
        for i in range(self.k):
            for r in self.sigma[1]:
                for op in self.op_r:
                    self.solver.add_clause([-(self.vars[X,op,r]+i)] + [self.vars[V,1,i]+j for j in range(i+1,self.k)])
            if NEG in self.op_b:
                self.solver.add_clause([-(self.vars[X,NEG]+i)] + [self.vars[V,1,i]+j for j in range(i+1,self.k)])
            for op in self.op_b - {NEG}:
                self.solver.add_clause([-(self.vars[X,op]+i)] + [self.vars[V,2,i]+j for j in range(i+1,self.k-1)])
            for j in range(self.k):
                for cn in self.sigma[0]:
                    self.solver.add_clause((-(self.vars[X,cn]+i),-(self.vars[Y,i]+j)))
                for b in {TOP,BOT}:
                    self.solver.add_clause((-(self.vars[X,b]+i),-(self.vars[Y,i]+j)))
            for j1 in range(self.k):
                for j2 in range(self.k):
                    if j1 != j2:
                        self.solver.add_clause((-(self.vars[Y,j1]+i),-(self.vars[Y,j2]+i)))                        

    def _evaluation_constraints(self):
        for a in range(self.A.max_ind):
            for i in range(self.k):                
                for op in self.op_b.difference({AND}):
                    self.solver.add_clause([-(self.vars[X,op]+i),-(self.vars[Z,a]+i)] + [ self.vars[U,op,i,a]+j for j in range(self.k) ])
                    for j in range(self.k):
                        self.solver.add_clause((-(self.vars[X,op]+i), self.vars[Z,a]+i, -(self.vars[U,op,i,a]+j)))
                if AND in self.op_b:
                    self.solver.add_clause([-(self.vars[X,AND]+i),self.vars[Z,a]+i] + [ -(self.vars[U,AND,i,a]+j) for j in range(self.k) ])
                    for j in range(self.k):
                        self.solver.add_clause((-(self.vars[X,AND]+i), -(self.vars[Z,a]+i), self.vars[U,AND,i,a]+j))
                
                if ALL in self.op_r:
                    for r in self.sigma[1]:
                        self.solver.add_clause([-(self.vars[X,ALL,r]+i), self.vars[Z,a]+i] + [ -(self.vars[U,ALL,i,b]+j) for j in range(self.k) for b in map(lambda x : x[0], filter(lambda t : t[1] == r , self.A.rn_ext[a])) ])                            
                        for j in range(self.k):
                            for b in map(lambda t : t[0],filter(lambda t : t[1] == r , self.A.rn_ext[a])):
                                self.solver.add_clause((-(self.vars[X,ALL,r]+i), -(self.vars[Z,a]+i), self.vars[U,ALL,i,b]+j))

                if EX in self.op_r:
                    for r in self.sigma[1]:
                        self.solver.add_clause([-(self.vars[X,EX,r]+i), -(self.vars[Z,a]+i)] + [ self.vars[U,EX,i,b]+j for j in range(self.k) for b in map(lambda x : x[0], filter(lambda t : t[1] == r , self.A.rn_ext[a]))])                        
                        for j in range(self.k):
                            for b in map(lambda t : t[0],filter(lambda t : t[1] == r , self.A.rn_ext[a])):
                                self.solver.add_clause((-(self.vars[X,EX,r]+i), self.vars[Z,a]+i, -(self.vars[U,EX,i,b]+j)))
                self.solver.add_clause((-(self.vars[X,TOP]+i),(self.vars[Z,a]+i)))
                self.solver.add_clause((-(self.vars[X,BOT]+i),-(self.vars[Z,a]+i)))
        for cn in self.sigma[0]:
            for i in range(self.k):
                for a in range(self.A.max_ind):                    
                    if a in self.A.cn_ext[cn]:                                            
                        self.solver.add_clause((-(self.vars[X,cn]+i), self.vars[Z,a]+i))
                    else:
                        self.solver.add_clause((-(self.vars[X,cn]+i),-(self.vars[Z,a]+i)))
                                    
    def _additional_constraints(self):
        if NEG in self.op:
            for i in range(self.k):
                for j in range(self.k):
                    for a in range(self.A.max_ind):
                        self.solver.add_clause((-(self.vars[U,NEG,i,a]+j), self.vars[Y,i]+j))
                        self.solver.add_clause((-(self.vars[U,NEG,i,a]+j), -(self.vars[Z,a]+j)))
                        self.solver.add_clause((self.vars[U,NEG,i,a]+j,-(self.vars[Y,i]+j), self.vars[Z,a]+j))
        if AND in self.op:
            for i in range(self.k):
                for j in range(self.k):
                    for a in range(self.A.max_ind):
                        self.solver.add_clause((self.vars[U,AND,i,a]+j, self.vars[Y,i]+j))
                        self.solver.add_clause((self.vars[U,AND,i,a]+j, -(self.vars[Z,a]+j)))
                        self.solver.add_clause((-(self.vars[U,AND,i,a]+j),-(self.vars[Y,i]+j), self.vars[Z,a]+j))
        if OR in self.op:
            for i in range(self.k):
                for j in range(self.k):
                    for a in range(self.A.max_ind):
                        self.solver.add_clause((-(self.vars[U,OR,i,a]+j), self.vars[Y,i]+j))
                        self.solver.add_clause((-(self.vars[U,OR,i,a]+j), self.vars[Z,a]+j))
                        self.solver.add_clause((self.vars[U,OR,i,a]+j,-(self.vars[Y,i]+j), -(self.vars[Z,a]+j)))
        if EX in self.op:
            for i in range(self.k):
                for j in range(self.k):
                    for a in range(self.A.max_ind):
                        self.solver.add_clause((-(self.vars[U,EX,i,a]+j), self.vars[Y,i]+j))
                        self.solver.add_clause((-(self.vars[U,EX,i,a]+j), self.vars[Z,a]+j))
                        self.solver.add_clause((self.vars[U,EX,i,a]+j,-(self.vars[Y,i]+j), -(self.vars[Z,a]+j)))
        if ALL in self.op:
            for i in range(self.k):
                for j in range(self.k):
                    for a in range(self.A.max_ind):
                        self.solver.add_clause((self.vars[U,ALL,i,a]+j, self.vars[Y,i]+j))
                        self.solver.add_clause((self.vars[U,ALL,i,a]+j, -(self.vars[Z,a]+j)))
                        self.solver.add_clause((-(self.vars[U,ALL,i,a]+j),-(self.vars[Y,i]+j), self.vars[Z,a]+j))

        for i in range(self.k):
            for j in range(self.k):
                li = list(range(self.k))
                li.remove(j)
                #self.solver.add_clause([self.vars[V,1,i]+j, -(self.vars[Y,i]+j)]+[self.vars[Y,i]+l for l in li])
                self.solver.add_clause((-(self.vars[V,1,i]+j), self.vars[Y,i]+j))                
                for l in li:
                    self.solver.add_clause((-(self.vars[V,1,i]+j),-(self.vars[Y,i]+l)))
        for i in range(self.k):
            for j in range(self.k-1):
                li = list(range(self.k))
                li.remove(j)
                li.remove(j+1)
                #self.solver.add_clause([self.vars[V,2,i]+j, -(self.vars[Y,i]+j+1),-(self.vars[Y,i]+j+1)]+[self.vars[Y,i]+l for l in li])
                self.solver.add_clause((-(self.vars[V,2,i]+j), self.vars[Y,i]+j))
                self.solver.add_clause((-(self.vars[V,2,i]+j), self.vars[Y,i]+j+1))
                for l in li:
                    self.solver.add_clause((-(self.vars[V,2,i]+j),-(self.vars[Y,i]+l)))

    def _fitting_constraints(self):
        for a in self.P:
            self.solver.add_clause([self.vars[Z,a]])            
        for a in self.N:
            self.solver.add_clause([-(self.vars[Z,a])])            

    def solve(self):
        #self._root()
        self._syn_tree_encoding()
        self._evaluation_constraints()
        self._additional_constraints()
        self._fitting_constraints()               
        if self.solver.solve():
           print("Satisfiable:")
           print(self._modelToTree())
           return True           
        else:
            print("Not satisfiable")
            return False
    
    def solve_incr(self,max_k,start_k=1):
        sat = False
        self.k = start_k
        while not sat and self.k <= max_k:            
            self.solver = Glucose4()
            self.vars = self._vars()
            self._syn_tree_encoding()
            self._evaluation_constraints()
            self._additional_constraints()
            self._fitting_constraints()               
            if self.solver.solve():
                print(f"Satisfiable for k={self.k}")
                print(self._modelToTree())
                sat = True
                return self.k, self._modelToTree()
            else:
                print(f"Not satisfiable for k={self.k}")             
                self.k += 1 
        return -1,""

    def printVariables(self):
        if self.solver.get_model():
            l = self.solver.get_model()                
            for k,v in self.vars.items():
                for i in range(self.k):
                    s = f"{bcolors.FAIL}False{bcolors.ENDC}"
                    if v+i in l:
                        s = f"{bcolors.OKGREEN}True{bcolors.ENDC}" 
                    if k[0] == X:
                        try:
                            print((d_var_names[k[0]],d_op[k[1]],v+i), f"Tree Node: {(v+i-1)%self.k }",s)
                        except KeyError:
                            print((d_var_names[k[0]],k[1],v+i), f"Tree Node: {(v+i-1)%self.k }",s)
                    elif k[0] == U:
                        print((d_var_names[k[0]],d_op[k[1]],f"domain element: {k[2]}",f"edge: ({k[3]},{i})",v+i),s)                
                    elif k[0] == V:
                        print((d_var_names[k[0]],k[1:],f"edge: ({k[2]},{i})",v+i),s)
                    else:
                        print((d_var_names[k[0]],k[1:],v+i),s)
                    if k[0] == Y:
                        print((d_var_names[k[0]],k[1:],v+i),s)
        
    def _modelToTree(self):
        m = self.solver.get_model()
        xv = sorted(list(filter(lambda x : x> 0, m[:self.vars[Y,0]-1])),key = lambda i : (i-1)%self.k)                
        edges = {i : [] for i in range(self.k)}
        x_symbols = [None] * self.k
        for x in m[:self.vars[Y,0]-1]:
            if x>0:
                i = (x-1)%(self.k)                
                x_symbols[i] = self.tree_node_symbols[x - i]
                for y in m[self.vars[Y,0]-1 + (i*self.k): self.vars[Y,0]-1 + (i*self.k) + self.k]:                
                    if y > 0:
                        j = ((y-1)%(self.k))
                        edges[i].append(j)                
        edges_labeled = { (k,x_symbols[k]) : list(map(lambda x : (x,x_symbols[x]), v)) for k,v in edges.items() }        
        t = STreeNode.FromDict(edges_labeled,(0,x_symbols[0]))        
        return t.to_asciitree()