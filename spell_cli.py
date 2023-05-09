import sys
import time
import argparse
from spell.fitting import solve_incr, mode

from spell.structures import solution2sparql, structure_from_owl


def main():
    parser = argparse.ArgumentParser(prog = 'SPELL')

    parser.add_argument("kb_owl_file", help="path to a OWL knowledge base in RDF/XML format")
    parser.add_argument("pos_example_list", help="path to a textfile containing positive examples")
    parser.add_argument("neg_example_list", help="path to a textfile containing negative examples")

    parser.add_argument("--max_size", type=int, default=19, help="(default=19)")
    parser.add_argument("--mode", choices=["exact", "neg_approx", "full_approx"], default=mode.exact, help="(default=exact)")
    parser.add_argument("--output", type=str, help="write best fitting SPARQL query to a file")
    parser.add_argument("--timeout", type=float, default=-1, help="in seconds (default=-1)")

    args = parser.parse_args()

    owlfile = args.kb_owl_file
    pospath = args.pos_example_list
    negpath = args.neg_example_list

    md = args.mode

    time_start = time.process_time()

    print("== Loading {}".format(owlfile))
    A, indmap, _ = structure_from_owl(owlfile)
    
    P: list[int] = []
    with open(pospath, encoding="UTF-8") as file:
        for line in file.readlines():
            ind = line.rstrip()
            if ind not in indmap:
                print("[ERR] The positive example {} does not seem to occur in {}".format(ind, owlfile))
                sys.exit(1)
            P.append(indmap[ind])

    N: list[int] = []
    with open(negpath, encoding="UTF-8") as file:
        for line in file.readlines():
            ind = line.rstrip()
            if ind not in indmap:
                print("[ERR] The negative example {} does not seem to occur in {}".format(ind, owlfile))
                sys.exit(1)
            N.append(indmap[ind])

    time_parsed = time.process_time()

    print("== Starting incremental search search for fitting queryy")
    time_start_solve = time.process_time()

    _, res  = solve_incr(A, P, N, md, timeout=args.timeout, max_size=args.max_size)

    time_solved = time.process_time()

    print(
        "== Took {:.2f}s for reading input and {:.3f}s for solving".format(
            time_parsed - time_start, time_solved - time_start_solve
        )
    )

    if args.output != None:
        print("== Writing result to {}".format(args.output))
        with open(args.output, "w", encoding="UTF-8") as file:
            file.write(solution2sparql(res))
        

if __name__ == "__main__":
    main()