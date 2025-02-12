import sys, time, json, os 
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy, Precision
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.class_expression import OWLClass
from ontolearn.refinement_operators import ModifiedCELOERefinement

from alc_benchmark import read_examples_from_json, instance_to_dllearner

def ontolearn_examples_to_dllearner(kb_path, ont_examples, dest, file_name_prefix):
    with open(ont_examples) as f:
        d = json.load(f)
        for p in d["problems"]:
            instance_to_dllearner(kb_path, d["problems"][p]["positive_examples"],d["problems"][p]["negative_examples"], os.path.join(dest,f"{file_name_prefix}_{p}"))

def ontolearn_examples_to_flat_json(ont_examples, dest):    
    with open(ont_examples) as f:
        d = json.load(f)        
        for p in d["problems"]:            
            dn = dict()
            dn["P"] = d["problems"][p]["positive_examples"]
            dn["N"] = d["problems"][p]["negative_examples"]
            dn["N_POS"] = len(dn["P"])
            dn["N_NEG"] = len(dn["N"])
            with open(os.path.join(dest,f"ol_ex_fam_rich_{p}.json"), "w+") as f:
                json.dump(dn, f)

def run(kb_path, P, N):
    start = time.time()
    kb = KnowledgeBase(path = kb_path)    
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, P)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, N)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
    end = time.time()
    kb_parse_time = end - start
    print(f"KB parsed after {kb_parse_time} seconds, starting CELOE next.")
    start = time.time()
    qual = Accuracy()
    heur = CELOEHeuristic(expansionPenaltyFactor=0.05, startNodeBonus=1.0, nodeRefinementPenalty=0.01)
    op = ModifiedCELOERefinement(knowledge_base=kb, use_negation=False, use_all_constructor=True)

    model = CELOE(knowledge_base=kb,
                  max_runtime=600,
                  refinement_operator=op,
                  quality_func=qual,
                  heuristic_func=heur,
                  max_num_of_concepts_tested=100,
                  iter_bound=100)
    model.fit(lp)
    hypotheses = list(model.best_hypotheses(n=3))    
    predictions = model.predict(individuals=list(typed_pos | typed_neg),
                                hypotheses=hypotheses)    
    [print(x) for x in hypotheses]
    end= time.time()
    print(f"Time for running CELOE: {end-start}")
    print(f"Total time: {end-start+kb_parse_time} seconds") 
    return 0,hypotheses[0]

def main():
    P,N = read_examples_from_json(sys.argv[2])
    run(sys.argv[1],P,N)
    #ontolearn_examples_to_dllearner(sys.argv[1], sys.argv[2])
    #ontolearn_examples_to_flat_json(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()