from rdflib import Graph
import sys
import pprint

from spell.fitting_alc import ALL, AND, EX, OR, FittingALC
from spell.structures import map_ind_name, structure_from_owl

#only female children
Q1 = """
SELECT DISTINCT ?0 WHERE {
    ?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
        FILTER NOT EXISTS {
            ?0 <http://schema.org/children> ?3.
                FILTER NOT EXISTS {
                    ?3 <http://schema.org/gender> ?4.
                    ?4 a <http://yago-knowledge.org/resource/Female__u0028_gender_u0029__class>.                    
                }
        }
    }
    LIMIT 25
"""

# at least one child and only female children
Q2 = """
SELECT DISTINCT ?0 WHERE {
    ?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
    ?0 <http://schema.org/children> ?2.
        FILTER NOT EXISTS {
            ?0 <http://schema.org/children> ?3.
                FILTER NOT EXISTS {
                    ?3 <http://schema.org/gender> ?4.
                    ?4 a <http://yago-knowledge.org/resource/Female__u0028_gender_u0029__class>.                    
                }
        }
    }
    LIMIT 25
""" 

#only female children
P1 = [
    'http://yago-knowledge.org/resource/Arndt_Bause',
    'http://yago-knowledge.org/resource/Radhanath_Rath',
    'http://yago-knowledge.org/resource/Rosita_Fornés'
    ]
#at least one male child
N1 = [
    'http://yago-knowledge.org/resource/Tony_Soprano',
    'http://yago-knowledge.org/resource/Konstantin_Rudanovsky'
    ]

def query(path, query):
    g = Graph()
    with open(path, 'r') as f:        
        g.parse(f, format='application/rdf+xml')
    qres = g.query(query)  
    for res in qres:
        print(res)

def test(path,P,N):
    A = structure_from_owl(path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A,6,P,N, op = {EX,ALL,OR,AND})    
    f.solve()

def main():
    #test(sys.argv[1], P1,N1)
    query(sys.argv[1],Q1)

if __name__ == "__main__":
    main()