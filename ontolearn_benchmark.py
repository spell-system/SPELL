import sys, time
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learners import CELOE
from ontolearn.heuristics import CELOEHeuristic
from ontolearn.learning_problem import PosNegLPStandard
from ontolearn.metrics import Accuracy, Precision
from owlapy.owl_individual import OWLNamedIndividual, IRI
from owlapy.class_expression import OWLClass
from ontolearn.refinement_operators import ModifiedCELOERefinement

from alc_benchmark import read_examples_from_json

def run(kb_path, P, N):
    start = time.time()
    kb = KnowledgeBase(path = kb_path)    
    typed_pos = set(map(OWLNamedIndividual, map(IRI.create, P)))
    typed_neg = set(map(OWLNamedIndividual, map(IRI.create, N)))
    lp = PosNegLPStandard(pos=typed_pos, neg=typed_neg)
    end = start = time.time()
    print(f"KB parsed after {end-start} seconds, starting CELOE next.")
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
    print(end-start)


def main():
    P,N = read_examples_from_json(sys.argv[2])
    run(sys.argv[1],P,N)

if __name__ == "__main__":
    main()