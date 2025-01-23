import sys, random, json, os, time
from rdflib import Graph
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
"""

# at least one male child
Q2 = """
SELECT DISTINCT ?0 WHERE {
    ?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
    ?0 <http://schema.org/children> ?1.
    ?1 <http://schema.org/gender> ?2.
    ?2 a <http://yago-knowledge.org/resource/Male_gender_class>.
    }
"""

# only female children or only male children 
Q3 = """
SELECT DISTINCT ?0 WHERE {    
    {?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
        FILTER NOT EXISTS {
            ?0 <http://schema.org/children> ?3.
                FILTER NOT EXISTS {
                    ?3 <http://schema.org/gender> ?4.
                    ?4 a <http://yago-knowledge.org/resource/Female__u0028_gender_u0029__class>.                    
                }
        }
    }
    UNION
        {?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
            FILTER NOT EXISTS {
                ?0 <http://schema.org/children> ?3.
                    FILTER NOT EXISTS {
                        ?3 <http://schema.org/gender> ?4.
                        ?4 a <http://yago-knowledge.org/resource/Male_gender_class>.                    
                    }
            }
        }
    }
    LIMIT 25
"""

#at least one male child and at least one female child
Q4 = """
SELECT DISTINCT ?0 WHERE {
    ?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
    ?0 <http://schema.org/children> ?1.
    ?1 <http://schema.org/gender> ?2.
    ?2 a <http://yago-knowledge.org/resource/Male_gender_class>.
    ?0 <http://schema.org/children> ?3.
    ?3 <http://schema.org/gender> ?4.
    ?4 a <http://yago-knowledge.org/resource/Female__u0028_gender_u0029__class>.    
}
LIMIT 10
"""

# at least one child and only female children
Q5 = """
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
"""

#at least one female child and only female children or at least one male child and only male children
Q6 = """
SELECT DISTINCT ?0 WHERE {    
    {?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
    ?0 <http://schema.org/children> ?5.
    ?5 <http://schema.org/gender> ?6.
    ?6 a <http://yago-knowledge.org/resource/Female__u0028_gender_u0029__class>.
        FILTER NOT EXISTS {
            ?0 <http://schema.org/children> ?3.
                FILTER NOT EXISTS {
                    ?3 <http://schema.org/gender> ?4.
                    ?4 a <http://yago-knowledge.org/resource/Female__u0028_gender_u0029__class>.                    
                }
        }
    }
    UNION
        {?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
        ?2 a <http://yago-knowledge.org/resource/Male_gender_class>.
        ?0 <http://schema.org/children> ?7.
        ?7 <http://schema.org/gender> ?8.
        ?8 a <http://yago-knowledge.org/resource/Male_gender_class>.
            FILTER NOT EXISTS {
                ?0 <http://schema.org/children> ?3.
                    FILTER NOT EXISTS {
                        ?3 <http://schema.org/gender> ?4.
                        ?4 a <http://yago-knowledge.org/resource/Male_gender_class>.                    
                    }
            }
        }
    }
"""

#no children
Q9 = """
SELECT DISTINCT ?0 WHERE {
    ?0 a <http://www.w3.org/2002/07/owl#NamedIndividual>.
    FILTER NOT EXISTS {
        ?0 <http://schema.org/children> ?3.
    }
}
"""

def query_and_print(path, query):
    g = Graph()
    with open(path, 'r') as f:        
        g.parse(f, format='application/rdf+xml')
    qres = g.query(query)  
    for res in list(qres):
        print(res[0].toPython())

def query_for_examples(path, q_pos, q_neg, n_pos, n_neg):
    g = Graph()
    with open(path, 'r') as f:        
        g.parse(f, format='application/rdf+xml')
    q_pos_res = g.query(q_pos)
    P = list(map(lambda x : x[0].toPython(), random.sample(list(q_pos_res),min(n_pos,len(q_pos_res)))))
    q_neg_res = g.query(q_neg)
    N = list(map(lambda x : x[0].toPython(), random.sample(list(q_neg_res),min(n_neg,len(q_neg_res)))))
    return P,N

def read_examples_from_json(path):
    with open(path) as f:
        o = json.load(f)
    return o["P"],o["N"]

def query_and_save(path, q_pos, q_neg, n_pos, n_neg, dest_path, filename):
    d = dict()
    P,N = query_for_examples(path, q_pos, q_neg, n_pos, n_neg)
    d["Q_POS"] = q_pos
    d["Q_NEG"] = q_pos
    d["N_POS"] = len(P)
    d["N_NEG"] = len(N)
    d["P"] = P
    d["N"] = N    
    with open(os.path.join(f"{dest_path}", f"{filename}.json"), 'w+') as f:
        json.dump(d,f, indent = 4)

def query_and_save(path, q_pos , q_neg, n_pos, n_neg, dest_path, filename):
    d = dict()
    P,N = query_for_examples(path, q_pos, q_neg, n_pos, n_neg)
    d["Q_POS"] = q_pos
    d["Q_NEG"] = q_pos
    d["N_POS"] = len(P)
    d["N_NEG"] = len(N)
    d["P"] = P
    d["N"] = N    
    with open(os.path.join(f"{dest_path}", f"{filename}.json"), 'w+') as f:
        json.dump(d,f, indent = 4)

def query_and_solve(path, q_pos, q_neg, n_pos, n_neg, k):
    P,N = query_for_examples(path, q_pos, q_neg, n_pos, n_neg)
    A = structure_from_owl(path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A,k,P,N, op = {EX,ALL,OR,AND})
    f.solve()

def solve(path, ex_path, k):
    A = structure_from_owl(path)
    P,N = read_examples_from_json(ex_path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A,k,P,N, op = {EX,ALL,OR,AND})    
    f.solve_incr(k)
    
def test(path,P,N):
    A = structure_from_owl(path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A,6,P,N, op = {EX,ALL,OR,AND})    
    f.solve()

def run_on_ontolearn_examples(kb_path, json_path, problem_key, k):
    A = structure_from_owl(kb_path)    
    with open(json_path) as f:
        d = json.load(f)
    P = d["problems"][problem_key]["positive_examples"]
    N = d["problems"][problem_key]["negative_examples"]
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    f = FittingALC(A,k,P,N, op = {EX,ALL,OR,AND})    
    f.solve_incr()

def main():
    start = time.time()
    #test(sys.argv[1], P1,N1)
    #query(sys.argv[1],Q9)
    #query_and_solve(sys.argv[1], Q3, Q4, 10,5,9)
    #query_and_save(sys.argv[1], Q6, Q4, 250,10, "alc_benchmarks","Q6Q4-250-10_nlp")
    solve(sys.argv[1],sys.argv[2], 30)
    #run_on_ontolearn_examples(sys.argv[1], sys.argv[2], "Grandgrandfather",6)
    end = time.time()
    print(f"Time: {end-start}")
if __name__ == "__main__":
    main()