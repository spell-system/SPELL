import sys, random, json, os, time
import pandas as pd, owlapy as ow
from pathlib import Path
from rdflib import Graph
from spell.benchmark_tools import construct_owl_from_structure
from spell.fitting_alc import ALL, AND, EX, OR, FittingALC
from spell.structures import map_ind_name, restrict_to_neighborhood, structure_from_owl
from owlready2 import default_world, get_ontology, owl
import ontolearn_benchmark


RANDOM_SEED = 42

random.seed(RANDOM_SEED)
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

Q10 = """
SELECT DISTINCT ?0 WHERE {?0 <http://schema.org/children> ?1.?0 male a <http://yago-knowledge.org/resource/Male_gender_class>.?0 <http://schema.org/children> ?1.?0 male a <http://yago-knowledge.org/resource/Male_gender_class>.?0 <http://schema.org/children> ?1.?0 female a <http://yago-knowledge.org/resource/Female__u0028_gender_u0029__class>.?0 <http://schema.org/children> ?1.?0 female a <http://yago-knowledge.org/resource/Female__u0028_gender_u0029__class>.}
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
    return f.solve_incr(k)
    
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
    f.solve_incr(k)

def instance_to_dllearner(kb_path, p, n, dest, file_name = "dl_instance"):
    file = os.path.join(dest,f"{file_name}.conf")
    with open(file, "w+") as f:
        f.write('ks.type = "OWL File"\n')
        f.write(f'ks.fileName = "{kb_path}"\n')
        f.write('reasoner.type = "closed world reasoner"\n')
        f.write('reasoner.sources = { ks }\n')
        f.write('lp.type = "posNegStandard"\n')
        f.write(f'lp.positiveExamples = {{{",".join(map(lambda x : f'"{x}"',p))}}}\n')
        f.write(f'lp.negativeExamples = {{{",".join(map(lambda x : f'"{x}"',n))}}}\n')
        f.write('alg.type = "celoe"\n')
        f.write('alg.maxExecutionTimeInSeconds = 1\n')
        f.write('alg.writeSearchTree = true\n')

def json_to_dllearner(kb_path, json_path, dest_dir):
    with open(json_path) as f:
        d = json.load(f)
        instance_to_dllearner(kb_path, d["P"] ,d["N"], dest_dir, Path(json_path).stem)

def jsons_to_dllearner(kb_path, dir, dest_dir):
    for file in os.listdir(dir):
        if os.path.splitext(file)[1] == '.json':
            json_to_dllearner(kb_path, os.path.join(dir, file), dest_dir)

def reduce_size_by_examples(kb_path, json_path, newpath, filename, k):
    P,N = read_examples_from_json(json_path)
    A = structure_from_owl(kb_path)
    P = list(map(lambda n: map_ind_name(A, n), P))
    N = list(map(lambda n: map_ind_name(A, n), N))
    B,m = restrict_to_neighborhood(k-1, A, P + N)
    construct_owl_from_structure(os.path.join(newpath,f"{filename}.owl"),B)

def qdepth(q_string):
    return 0

def benchmark(kb_path,queries_path, dest_dir):
    cols = ["kb", "kb_file_reduced", "n_pos", "n_neg" , "run", "alc sat time", "alc sat k", "alc sat concept", "ontolearn celoe time", "ontolearn celoe accuracy", "ontolearn celoe concept"]    
    p_avg = pd.DataFrame(columns=cols)
    data = []
    data_avg = []
    os.mkdir(dest_dir)
    for q_pos,q_neg in [
       ("Q1","Q4"),
        ("Q3","Q5"),
        ("Q6", "Q7")
        #("Q12","Q13")
    ]:
        with open(queries_path) as f:
            d = json.load(f)
        k = max(d[q_pos]["depth"],d[q_neg]["depth"])
        k = 30
        for n_pos in [10,100,1000]:
            for n_neg in [10,100,1000]:
                avg_time_spell_alc_sum = 0
                avg_time_celoe_sum = 0            
                avg_accuracy_sum = 0
                red_kb_path_filename = f"reduced_kb-pos_({q_pos},{n_pos})-neg_({q_neg},{n_neg})-k_{k}"
                red_kb_path = os.path.join(dest_dir,f"{red_kb_path_filename}.owl")
                js_path = os.path.join(dest_dir,f"{red_kb_path_filename}.json")
                if not examples_by_queries(kb_path, queries_path, q_pos,q_neg, n_pos, n_neg, dest_dir, f"{red_kb_path_filename}.json" ):
                    break
                reduce_size_by_examples(kb_path,js_path,dest_dir,red_kb_path_filename,k)
                alc_k = 0
                for i in range(1,11):
                    start = time.time()
                    (l,c_alcsat) = solve(red_kb_path, js_path, k)
                    alc_k += l
                    end = time.time()
                    alc_time = end -start
                    avg_time_spell_alc_sum += alc_time
                    P,N = read_examples_from_json(js_path)
                    start = time.time()
                    a,c_celoe = ontolearn_benchmark.run(red_kb_path,P,N)
                    end = time.time()
                    avg_accuracy_sum += a
                    celoe_time = end-start
                    avg_time_celoe_sum += celoe_time
                    data.append([kb_path,red_kb_path_filename, n_pos, n_neg, i,alc_time, l, c_alcsat, celoe_time, a, c_celoe])
                data.append([kb_path,red_kb_path_filename, n_pos, n_neg, "avg" ,avg_time_spell_alc_sum / 10, alc_k/10, "", avg_time_celoe_sum/10, avg_accuracy_sum/10, ""])
                data_avg.append([kb_path,red_kb_path_filename, n_pos, n_neg, "avg" ,avg_time_spell_alc_sum / 10, alc_k/10, "", avg_time_celoe_sum/10, avg_accuracy_sum/10, ""])
                
    p = pd.DataFrame(data,columns=cols)
    pa = pd.DataFrame(data_avg,columns=cols)
    p.to_csv(os.path.join(dest_dir, "results.csv"))
    pa.to_csv(os.path.join(dest_dir, "results_avg.csv"))

def examples_by_queries(kb_path, queries_path, q_pos, q_neg, n_pos, n_neg, dest_dir, file_name , ensure_no_contradiction = True, random_pos = True, random_neg = True):
    g = get_ontology(kb_path).load()
    d = dict()
    d["q_pos"] = q_pos
    d["q_neg"] = q_neg
    d["n_pos"] = n_pos
    d["n_neg"] = q_neg    
    with open(queries_path, 'r') as f:
        d = json.load(f)
    p_res = list( map(lambda x : x[0].get_iri(), default_world.sparql(d[q_pos]["SPARQL"])))
    if not p_res:
        return False
    if random_pos:
        P = random.sample(p_res,n_pos)
    else:
        P = p_res[:n_pos]
    n_res = list(map(lambda x : x[0].get_iri(), default_world.sparql(d[q_neg]["SPARQL"])))
    if not n_res:
        return False
    if random_neg:
        N = random.sample(n_res,n_neg)
    else:
        N = n_res[:n_pos]
    d["P"] = P
    d["N"] = N
    d["rnd_pos"] = True
    d ["end_neg"] = False
    d["random_seed"] = RANDOM_SEED
    with open(os.path.join(dest_dir,file_name), "w+") as f:
        json.dump(d,f)
    return True

def main():
    start = time.time()
    #test(sys.argv[1], P1,N1)
    #query(sys.argv[1],Q9)
    #query_and_solve(sys.argv[1], Q3, Q4, 10,5,9)
    #query_and_save(sys.argv[1], Q1, Q2, 25,10, sys.argv[2],"Q1Q2-25-10_nl")
    #solve(sys.argv[1],sys.argv[2], 30)
    #run_on_ontolearn_examples(sys.argv[1], sys.argv[2], "Cousin",12)
    #jsons_to_dllearner(sys.argv[1],sys.argv[2],sys.argv[3])
    #reduce_size_by_examples(sys.argv[1], sys.argv[2], 20)
    #examples_by_queries(sys.argv[1],sys.argv[2],"Q1", "Q2", 10,5,"", "")
    benchmark(sys.argv[1],sys.argv[2], sys.argv[3])

if __name__ == "__main__":
    main()