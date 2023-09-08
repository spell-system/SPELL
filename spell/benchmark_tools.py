import functools
import os
import random
import time
from typing import FrozenSet, Generator, Union

from .fitting import non_empty_symbols, solve_incr, mode
from .structures import (ABoxBuilder, Signature, Structure, conceptname_ext,
                         conceptnames, copy_structure, generate_all_trees, ind,
                         rolenames, solution2sparql, structure_from_owl)

ROBOT_PATH = "{}/../robot/robot".format(os.path.dirname(os.path.realpath(__file__)))
ROBOT_JAVA_ARGS = "-Xmx40G"

Concept = FrozenSet[tuple[str, Union[None, "Concept"]]]


def drop_leave_atom(c: Concept) -> list[Concept]:
    res: list[Concept] = []

    for (rn, d) in c:
        weak_c = c - {(rn, d)}
        if not d:
            res.append(weak_c)
        elif len(d) == 0:
            res.append(weak_c)
        else:
            for weak_d in drop_leave_atom(d):
                res.append(weak_c | {(rn, weak_d)})

    return res


def weaken_concept(concept: Concept, steps: int) -> set[Concept]:
    s = {concept}

    for _ in range(steps):
        s = {d for c in s for d in drop_leave_atom(c)}

    return s


def add_random_cn(A: Structure, sigma: Signature) -> Structure:
    a = random.randrange(A[0])
    cn = random.choice(conceptnames(sigma))

    attempts = 0
    while a in conceptname_ext(A, cn) and attempts < 100:
        a = random.randrange(A[0])
        cn = random.choice(conceptnames(sigma))
        attempts += 1

    A2 = copy_structure(A)

    A2[1][cn].add(a)
    return A2


def random_concept(size: int, sigma: Signature) -> Structure:
    edges = 0
    concept_assertions = 100000

    # Make sure that we don't have more concept assertions than ways to apply concept assertions
    while (
        concept_assertions > (edges + 1) * len(conceptnames(sigma))
        or concept_assertions < edges + 1
    ):
        edges = random.randrange(int(size * 0.80))  # 40% edges, 60% concept assertions
        concept_assertions = size - edges

    all_trees = list(generate_all_trees(edges + 1))
    tree = random.choice(all_trees)
    assert len(tree) == edges

    rasserts: dict[int, set[tuple[int, str]]] = {}
    for i in range(size):
        rasserts[i] = set()

    for edge in range(edges):
        rolename = random.choice(rolenames(sigma))
        rasserts[tree[edge]].add((edge + 1, rolename))

    casserts = {}
    for cn in conceptnames(sigma):
        casserts[cn] = set()

    # Generate distinct concept assertions
    no_casserts = 0
    while no_casserts < concept_assertions:
        cn = random.choice(conceptnames(sigma))
        ind = random.randrange(edges + 1)
        if ind not in casserts[cn]:
            casserts[cn].add(ind)
            no_casserts += 1

    return (edges + 1, casserts, rasserts)


def frontier(c: Concept) -> list[Concept]:
    res: list[Concept] = []

    for (rn, d) in c:
        base: Concept = c - {(rn, d)}
        if not d:  # Conceptname
            res.append(base)
        elif len(d) == 0:  # Leaf
            res.append(base)
        else:
            for fd in frontier(d):
                base = base | {(rn, fd)}
            res.append(base)
        continue

    return res

def repeated_frontier(c: Concept, n: int) -> list[Concept]:
    f = [c]

    for i in range(n):
        f = list({ c for d in f for c in frontier(d)})

    return f

def drop_root_subtree(c: Concept) -> list[Concept]:
    res: list[Concept] = []
    for (rn, d) in c:
        res.append(c - {(rn, d)})
    return res


def weaken_drop_root_subtrees(c: Concept, succs: int) -> list[Concept]:
    r1 = frontier(c)

    while len(r1[0]) > succs:
        r1 = list(set([d for c in r1 for d in drop_root_subtree(c)]))

    return list(set(r1))


def concept2sparqlclauses(concept: Concept, counter) -> list[str]:
    res: list[str] = []

    thisnode = counter
    if counter == 0:
        res.append("?{} a <http://www.w3.org/2002/07/owl#Thing> .".format(counter))
    if len(concept) > 0:
        res.append("FILTER EXISTS {")
        for (rn, d) in concept:
            if d is None:
                res.append("?{} a {} .".format(thisnode, rn))
            else:
                res.append("?{} {} ?{} .".format(thisnode, rn, counter + 1))
                sub = concept2sparqlclauses(d, counter + 1)
                res.extend(sub)
                counter += len(sub) + 1
        res.append("}")

    return res


def concept2sparql(concept: Concept) -> str:
    clauses = concept2sparqlclauses(concept, 0)

    return "SELECT DISTINCT ?0 WHERE {{\n {}\n}}".format("\n ".join(clauses))


def sparql2struct(sparql: str) -> Structure:
    parts = sparql.split(",")

    b = ABoxBuilder()

    for assertion in parts[1 : len(parts) - 1]:
        ass_parts = assertion.split(" ")
        ind = b.map_ind(ass_parts[1])
        if ass_parts[2] == "a":
            b.concept_assertion(ind, ass_parts[3])
        else:
            b.role_assertion(ind, ass_parts[3], ass_parts[2])

    return b.A


def conj2string(rn: str, d: Union[None, Concept]) -> str:
    if d is None:
        return "{}".format(rn)
    if len(d) > 1:
        return "\\exists {}.({})".format(rn, concept2string(d))
    else:
        return "\\exists {}.{}".format(rn, concept2string(d))


def concept2string(concept: Concept) -> str:
    if len(concept) == 0:
        return "\\top"
    sub_concepts = list(conj2string(rn, d) for (rn, d) in concept)
    sub_concepts.sort()
    return " \\sqcap ".join(sub_concepts)


@functools.cache
def number_of_vars(c: Union[None, Concept]) -> int:
    if c is None:
        return 0
    return 1 + sum([number_of_vars(d) for (rn, d) in c])


def concept_depth(c: Union[None, Concept]) -> int:
    if c is None:
        return 0
    if len(c) == 0:
        return 0
    return 1 + max([concept_depth(d) for (rn, d) in c])


def structure2concept_rec(s: Structure, i: int) -> Concept:
    res: Concept = frozenset()
    for cn in s[1].keys():
        if i in s[1][cn]:
            res = res | {(cn, None)}

    for (j, r) in s[2][i]:
        c2 = structure2concept_rec(s, j)
        res = res | {(r, c2)}

    return res


def structure2concept(s: Structure) -> Concept:
    return structure2concept_rec(s, 0)


def concept2structure(c: Concept) -> Structure:
    queue: list[tuple[Concept, int]] = [(c, 0)]

    res: Structure = (1, {}, {})

    while len(queue) > 0:
        (c, ind) = queue.pop(0)
        res[2][ind] = set()

        for (r, c2) in c:
            if c2 == None:
                if r not in res[1].keys():
                    res[1][r] = set()
                res[1][r].add(ind)
            else:
                ind2 = res[0]
                res = (res[0] + 1, res[1], res[2])
                res[2][ind].add((ind2, r))
                queue.append((c2, ind2))

    return res


def get_reachable_inds(owlfile, starts: list[str]) -> list[str]:
    A, indmap, nsmap = structure_from_owl(owlfile)
    new_elems = {indmap[s] for s in starts}
    res: set[int] = set()

    while len(new_elems) > 0:
        res |= new_elems

        new_new_elems: set[int] = set()
        for a in new_elems:
            for (b, r) in A[2][a]:
                if b not in res:
                    new_new_elems.add(b)

        new_elems = new_new_elems

    relevant_inds: list[str] = []
    for (k, v) in indmap.items():
        # Restrict to http since output needs to be IRIs
        # This also drops "Fresh" individuals, introduced by TBox reasoning
        if "http" in k and v in res:
            relevant_inds.append(k)

    return relevant_inds


def run_robot_cmd(cmd: str):
    if not os.path.isfile(ROBOT_PATH):
        print("robot cmd at {} not found".format(ROBOT_PATH))

    import subprocess

    my_env = os.environ.copy()
    my_env["ROBOT_JAVA_ARGS"] = ROBOT_JAVA_ARGS

    subprocess.run(cmd, shell=True, check=True, env=my_env)


def create_restricted_owl(owlfile, individuals: list[str], result):
    tmp_file = ".filter-inds.txt"

    with open(tmp_file, "w") as file:
        for a in individuals:
            file.write(a + "\n")

    cmd = '{} \
     reason --input {} --axiom-generators "ClassAssertion" --include-indirect true \
     remove --term-file {} --select complement --select individuals \
     remove --select individuals --select types --select complement --select classes \
     remove --term "rdfs:label" remove --term "rdfs:comment" \
    --output {}'

    fullcmd = cmd.format(ROBOT_PATH, owlfile, tmp_file, result)

    run_robot_cmd(fullcmd)


def create_materialized_tdb_dir(owlfile, tdb_dir):
    tmp_owl = "tmp.owl"

    if os.path.isdir("./" + tdb_dir):
        import shutil

        shutil.rmtree("./" + tdb_dir)

    A, indmap, nsmap = structure_from_owl(owlfile)
    construct_owl_from_structure(tmp_owl, A, indmap, nsmap)

    cmd = "{} query --input {} --create-tdb true --tdb-directory {}"
    fullcmd = cmd.format(ROBOT_PATH, tmp_owl, tdb_dir)
    run_robot_cmd(fullcmd)


def owlname2tdbname(owlfile):
    return ".cache/{}".format(owlfile.replace("/", "-"))


def parse_query_output(output_file) -> list[str]:
    result: list[str] = []
    with open(output_file) as file:
        try:
            next(file)  # skip first line
        except StopIteration:
            # File is empty = no answers
            return []
        for line in file:
            line = line.replace('"', "")
            if "%" in line:
                continue  # DL-Learner has issues deling with url encoded stuff, so we ignore these results
            if line.rstrip():
                result.append(line.rstrip())
    result.sort()
    return result


@functools.cache
def query_tdbdir(tdbdir, sparql) -> list[str]:
    filename = ".cl.sparql"
    output = ".out.csv"
    cmd = "{} query --tdb-directory {} --keep-tdb-mappings true --query {} {}"

    print(sparql)

    with open(filename, "w") as file:
        file.write(sparql)

    fullcmd = cmd.format(ROBOT_PATH, tdbdir, filename, output)

    run_robot_cmd(fullcmd)

    return parse_query_output(output)


def query_owl(owlfile, sparql, limit=0) -> list[str]:
    filename = ".cl.sparql"
    output = ".out.csv"

    print(sparql)

    with open(filename, "w") as file:
        file.write(sparql)

    cmd = "{} query --input {} --query {} {}"

    fullcmd = cmd.format(ROBOT_PATH, owlfile, filename, output)

    run_robot_cmd(fullcmd)

    return parse_query_output(output)


def merge_negatives(negs: list[list[str]]):
    import itertools

    # Should be a round-robin merge of all negative examples

    return [x for x in itertools.chain(*itertools.zip_longest(*negs)) if x is not None]


def query_for_benchmark_examples(
    tdbdir: str, concept: Concept, steps: int, bound: int
) -> tuple[list[str], list[str]]:
    ws = repeated_frontier(concept, steps)

    positive = query_tdbdir(tdbdir, concept2sparql(concept))
    positive.sort()

    negative = []
    for w in ws:
        res = query_tdbdir(tdbdir, concept2sparql(w))
        res = list(set(res) - set(positive))
        res.sort()

        negative.append(res)

    negative = list(
        set(merge_negatives(negative))
    )  # deduplicate, we lose the ordering though, can we improve that?
    negative.sort()

    return positive[0:bound], negative[0:bound]


def emit_sml_benchmark(
    path: str, name: str, owlfile: str, P: list[str], N: list[str], info: list[str]
) -> None:
    import subprocess

    print("== Creating benchmark directory at {}/{}".format(path, name))
    example_dir = "{}/{}/owl/lp/1".format(path, name)
    p_example_path = "{}/pos.txt".format(example_dir)
    n_example_path = "{}/neg.txt".format(example_dir)
    dll_conf_path = "{}/dllearner.conf".format(example_dir)
    info_path = "{}/{}/benchmark-info.txt".format(path, name)

    os.makedirs(example_dir, exist_ok=True)

    with open(p_example_path, "w") as the_file:
        for p in P:
            the_file.write(p + "\n")

    with open(n_example_path, "w") as the_file:
        for n in N:
            the_file.write(n + "\n")

    with open(dll_conf_path, "w") as file:
        file.write("[main]\n")
        file.write("loadingTime=30\n")
        file.write("algorithm.type = \"eltl\"\n")
        file.write("algorithm.stopOnFirstDefinition = true\n")
        if "owl2bench" not in name:
            file.write("algorithm.useMinimizer = true\n")
        else:
            file.write("algorithm.useMinimizer = false\n")
        file.write("algorithm.maxClassExpressionDepth = 16\n")


    with open(info_path, "w") as file:
        file.write("Benchmark is automatically generated for SPELL\n")
        file.write("Number of positive examples: {}\n".format(len(P)))
        file.write("Number of negative examples: {}\n".format(len(N)))

        for v in info:
            file.write(v + "\n")

    owl_dir = "{}/{}/owl/data".format(path, name)
    owl_path = "{}/{}.owl".format(owl_dir, name)

    os.makedirs(owl_dir, exist_ok=True)

    subprocess.run("cp {} {}".format(owlfile, owl_path), shell=True)


def construct_sml_benchmark(
    path, name, owlfile, concept, weaken_steps=1, size_bound=50
):
    print("== Generating benchmark {}".format(name))
    print("== Saturating {} and creating cache for querying".format(owlfile))

    tdbdir = owlname2tdbname(owlfile)

    create_materialized_tdb_dir(owlfile, tdbdir)

    print("== Querying {} using {}".format(owlfile, ROBOT_PATH))
    P, N = query_for_benchmark_examples(tdbdir, concept, weaken_steps, size_bound)

    tmp_owl = "tmp.owl"

    print("== Collecting relevant individuals for this benchmark")
    relevant_inds = get_reachable_inds(owlfile, list(P) + list(N))

    print("== Creating relevant subset of {}".format(owlfile))
    create_restricted_owl(owlfile, relevant_inds, tmp_owl)

    emit_sml_benchmark(
        path,
        name,
        tmp_owl,
        P,
        N,
        [
            "Fragment of knowledge base: {}".format(owlfile),
            "Total number of individuals: {}".format(len(relevant_inds)),
            "Target query: {}".format(concept2string(concept)),
            "Generalization steps: {}".format(weaken_steps),
        ],
    )

    print(
        "== Successfully generated benchmark {} with {} + {} examples and {} individuals".format(
            name, len(P), len(N), len(relevant_inds)
        )
    )


def parse_simple_concept(concept_str: list[str]) -> tuple[list[str], Concept]:
    if concept_str[0] == "\\exists":
        rn = concept_str[1]
        if concept_str[2] == "(":
            concept_str, res = parse_conjunction(concept_str[3:])
            assert concept_str[0] == ")"
            return concept_str[1:], frozenset({(rn, res)})
        else:
            concept_str, res = parse_simple_concept(concept_str[2:])
            return concept_str, frozenset({(rn, res)})
    if concept_str[0] == "\\top":
        return concept_str[1:], frozenset()
    return concept_str[1:], frozenset({(concept_str[0], None)})


def parse_conjunction(concept_str: list[str]) -> tuple[list[str], Concept]:
    concept_str, res = parse_simple_concept(concept_str)
    if len(concept_str) > 0:
        if concept_str[0] == ")":
            return concept_str, res
        if concept_str[0] == "\\sqcap":
            concept_str, res2 = parse_conjunction(concept_str[1:])
            res = res | res2
    return concept_str, res


def parse_concept(concept_str: str) -> Concept:
    # concept_str = concept_str.replace(".", " ")
    concept_str = concept_str.replace("(", " ( ")
    concept_str = concept_str.replace(")", " ) ")
    concept_str = concept_str.replace("  ", " ")
    concept_str = concept_str.strip()
    b = concept_str.split(" ")
    b, res = parse_conjunction(b)
    assert len(b) == 0
    return res


def verify_solution(owlfile, P, N, indmap, solution):
    claimed_acc, best_q = solution

    print("== Querying {} with best solution".format(owlfile))

    A, indmap, nsmap = structure_from_owl(owlfile)
    construct_owl_from_structure("tmp.owl", A, indmap, nsmap)

    result = query_owl("tmp.owl", solution2sparql(best_q))

    rs = set()
    for res in result:
        if res in indmap:
            rs.add(indmap[res])

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for p in P:
        if p in rs:
            tp += 1
        else:
            fn += 1

    for n in N:
        if n in rs:
            fp += 1
        else:
            tn += 1

    real_acc = tp + tn

    print("== Real accuracy {}/{} {}/{}".format(tp, len(P), tn, len(N)))
    assert real_acc == claimed_acc


def load_sml_tasks(path: str, task: str):
    basepath = "{}/{}".format(path, task)
    owlpath = "{}/owl/data/{}.owl".format(basepath, task)

    print("== Loading {} for benchmark {}".format(owlpath, task))
    A, indmap, _ = structure_from_owl(owlpath)

    res: dict[str, tuple[str, Structure, list[int], list[int], dict[str, int]]] = {}
    for lp in os.listdir("{}/owl/lp".format(basepath)):

        pospath = "{}/owl/lp/{}/pos.txt".format(basepath, lp)
        negpath = "{}/owl/lp/{}/neg.txt".format(basepath, lp)

        with open(pospath, encoding="UTF-8") as file:
            P = [indmap[line.rstrip()] for line in file.readlines()]

        with open(negpath, encoding="UTF-8") as file:
            N = [indmap[line.rstrip()] for line in file.readlines()]

        res[lp] = (owlpath, A, P, N, indmap)
    return res


def generate_benchmark_collection(path, prefix, owlfile, concepts, size_bound: int):
    print("== Generating benchmark {}".format(prefix))
    print("== Saturating {} and creating cache for querying".format(owlfile))

    tdbdir = owlname2tdbname(owlfile)

    create_materialized_tdb_dir(owlfile, tdbdir)

    relevant_inds = set()
    benchmarks = set()

    total_queries = sum([1 + len(nCs) for (info, pC, nCs) in concepts])
    current_query = 1

    examples = {}
    for (info, pC, nCs) in concepts:
        name = "{}-{}-{}".format(prefix, info, size_bound)
        benchmarks.add(name)

        print("Query {}/{}".format(current_query, total_queries))
        P = query_tdbdir(tdbdir, concept2sparql(pC))
        current_query += 1
        relevant_inds |= set(P[0:size_bound])
        Ns = []
        for nC in nCs:
            print("Query {}/{}".format(current_query, total_queries))
            N = query_tdbdir(tdbdir, concept2sparql(nC))
            current_query += 1
            N = list(set(N) - set(P))
            N.sort()
            Ns.append(N)

        N = list(
            set(merge_negatives(Ns))
        )  # deduplicate, we lose the ordering though, can we improve that?
        N.sort()
        relevant_inds |= set(N[0:size_bound])
        examples[name] = (P[0:size_bound], N[0:size_bound])

    print("== Collecting reachable individuals")
    relevant_inds = get_reachable_inds(owlfile, list(relevant_inds))

    print("== Creating reachable fragment of {}".format(owlfile))
    tmp_owl = "temp.owl"
    create_restricted_owl(owlfile, relevant_inds, tmp_owl)

    for benchmark in benchmarks:
        emit_sml_benchmark(
            path, benchmark, tmp_owl, examples[benchmark][0], examples[benchmark][1], []
        )

    print(
        "== Successfully generated benchmark collection {} with {} benchmarks".format(
            prefix, len(concepts)
        )
    )


@functools.cache
def encode(s) -> str:
    parts = s.split("/")
    parts[-1] = parts[-1].replace("&", "&amp;")
    return "/".join(parts)


@functools.cache
def class_string(cn: str) -> str:
    return '    <rdf:type rdf:resource="{}"/>\n'.format(encode(cn))


def construct_owl_from_structure(
    filename, A: Structure, indmap: dict[str, int], nsmap: dict[str | None, str]
):
    sigma: Signature = non_empty_symbols(A)

    reverse_indmap = {
        n: name for (name, n) in indmap.items() if "#" in name or "/" in name or "NC_" in name
    }
    reverse_nsmap = {ns: key for (key, ns) in nsmap.items() if key != None}

    rev_cns = {a: set() for a in ind(A)}
    for cn in conceptnames(sigma):
        for a in conceptname_ext(A, cn):
            rev_cns[a].add(cn)

    with open(filename, "w") as file:
        file.write('<?xml version="1.0"?> \n <rdf:RDF ')
        for (key, ns) in nsmap.items():
            if key == None:
                file.write('    xmlns="{}"\n'.format(ns))
            else:
                file.write('    xmlns:{}="{}"\n'.format(key, ns))
        file.write(">\n")
        file.write(
            '<owl:Ontology rdf:about="{}"/>\n'.format(nsmap[None].replace("#", ""))
        )

        for cn in conceptnames(sigma):
            file.write('<owl:Class rdf:about="{}"/>\n'.format(encode(cn)))

        for rn in rolenames(sigma):
            file.write('<owl:ObjectProperty rdf:about="{}"/>\n'.format(encode(rn)))

        for a in ind(A):
            file.write(
                '<owl:NamedIndividual rdf:about="{}">\n'.format(
                    encode(reverse_indmap[a])
                )
            )

            for cn in rev_cns[a]:
                file.write(class_string(cn))

            for (b, r) in A[2][a]:
                for (ns, key) in reverse_nsmap.items():
                    r = r.replace(ns, "{}:".format(key))
                if nsmap[None] in r:
                    r = r.replace(nsmap[None], "")

                file.write(
                    '    <{} rdf:resource="{}"/>\n'.format(r, encode(reverse_indmap[b]))
                )

            file.write("</owl:NamedIndividual>\n")

        file.write("</rdf:RDF>\n")


def construct_owl_from_concepts(
    filename, ps: list[Concept], ns: list[Concept]
) -> tuple[list[str], list[str]]:

    pos_inds: list[str] = []
    neg_inds: list[str] = []

    sigma: Signature = ([], [])
    for p in ps:
        sign = non_empty_symbols(concept2structure(p))

        sigma = (list(set(sigma[0]) | set(sign[0])), list(set(sigma[1]) | set(sign[1])))

    with open(filename, "w") as file:
        file.write(
            '<?xml version="1.0"?> \n <rdf:RDF xmlns="urn:absolute:test#" xml:base="urn:absolute:test" xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:xsd="http://www.w3.org/2001/XMLSchema#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:test="http://example.com/test#"> <owl:Ontology rdf:about="urn:absolute:test"/> \n'
        )

        for cn in conceptnames(sigma):
            file.write(
                '<owl:Class rdf:about="http://example.com/test#{}"/>\n'.format(cn)
            )

        for rn in rolenames(sigma):
            file.write(
                '<owl:ObjectProperty rdf:about="http://example.com/test#{}"/>\n'.format(
                    rn
                )
            )

        maxind = 0
        queue: list[tuple[Concept, int]] = []
        for p in ps:
            queue.append((p, maxind))
            pos_inds.append("http://example.com/test#a{}".format(maxind))
            maxind += 1

        for n in ns:
            queue.append((n, maxind))
            neg_inds.append("http://example.com/test#a{}".format(maxind))
            maxind += 1

        while len(queue) > 0:
            c, i = queue.pop(0)

            file.write(
                '<owl:NamedIndividual rdf:about="http://example.com/test#a{}">\n'.format(
                    i
                )
            )

            for (r, c2) in c:
                if c2 == None:
                    file.write(
                        '    <rdf:type rdf:resource="http://example.com/test#{}"/>\n'.format(
                            r
                        )
                    )
                if c2 != None:
                    file.write(
                        '    <test:{} rdf:resource="http://example.com/test#a{}"/>\n'.format(
                            r, maxind
                        )
                    )
                    queue.append((c2, maxind))
                    maxind += 1

            file.write("</owl:NamedIndividual>\n")

        file.write("</rdf:RDF>\n")

        return (pos_inds, neg_inds)


def parse_eltl_paren(parts: list[str]) -> tuple[Concept, list[str]]:
    if parts[0] != "(":
        return parse_eltl_simple_concept(parts)
    else:
        C, parts = parse_eltl_conj(parts[1:])
        assert parts[0] == ")"
        return C, parts[1:]


def parse_eltl_simple_concept(parts: list[str]) -> tuple[Concept, list[str]]:
    assert len(parts) > 0

    if len(parts) > 1 and parts[1] == "some":
        rn = parts[0]

        if parts[2] == "(":

            C, parts = parse_eltl_conj(parts[3:])

            assert parts[0] == ")"
            # existential
            return frozenset({(rn, C)}), parts[1:]
        else:
            C, parts = parse_eltl_conj(parts[2:])

            # existential
            return frozenset({(rn, C)}), parts[0:]
    else:
        C = parts[0]
        if C == "Thing":
            return frozenset(), parts[1:]
        return frozenset({(C, None)}), parts[1:]


def parse_eltl_conj(parts: list[str]) -> tuple[Concept, list[str]]:
    C, parts = parse_eltl_paren(parts)

    while len(parts) > 0 and (parts[0] == "and" or parts[0] == "or"):
        C2, parts = parse_eltl_paren(parts[1:])
        C = C | C2

    return C, parts


@functools.cache
def cn_signature(c: Concept) -> set[str]:
    res = set()

    for (rn, c1) in c:
        res.add(rn)
        if c1 != None:
            res |= cn_signature(c1)
    return res


# is d stronger than c
@functools.cache
def subsum(c: Concept, d: Concept) -> bool:
    for (rn, c1) in c:
        if c1 == None:  # Conceptname
            if (rn, None) not in d:
                return False
        else:
            found = False
            for (rn2, d1) in d:
                if rn != rn2:
                    continue
                if subsum(c1, d1):
                    found = True
                    break
            if not found:
                return False

    return True


def is_addition_still_core(base: Concept, rn, add) -> bool:
    if not cn_signature(add).issubset(cn_signature(base)):
        return True
    for (rn2, d) in base:
        if d == None or rn2 != rn:
            continue
        if subsum(add, d):
            return False  # d is already stronger than add

    return True


def core_frontier(c: Concept) -> Generator[Concept, None, None]:
    for (rn, d) in c:
        base: Concept = c - {(rn, d)}
        if not d:  # Conceptname
            yield base
        elif len(d) == 0:  # Leaf
            yield base
        else:
            fg = core_frontier(d)
            res = base
            for fd in fg:
                if is_addition_still_core(base, rn, fd):
                    res = res | {(rn, fd)}
            yield res


# Naive implementation
def distance_from_top(c: Concept) -> int:
    largest = 0
    res = 0
    while len(c) > 0:
        res += 1
        g = core_frontier(c)
        c = g.__next__()
        sz = number_of_vars(c)
        if sz > largest:
            print("{} {}".format(res, number_of_vars(c)))
            largest = sz
        if res > 10000:
            return res
    return res


def parse_eltl(c: str) -> Concept:
    c = c.replace("(", " ( ")
    c = c.replace(")", " ) ")
    c = c.replace("  ", " ")
    c = c.strip()

    parts = c.split(" ")

    C, parts = parse_eltl_conj(parts)

    assert len(parts) == 0
    return C


def labeled_r_path_dual(path: list[str], cns: set[str]) -> Concept:
    if len(path) == 0:
        return frozenset()
    c = path[0]
    rest = labeled_r_path_dual(path[1:], cns)
    if len(path) > 1:
        rest = rest | {(cn, None) for cn in cns}

    attach = frozenset({("r", labeld_r_path(len(path) - 1, cns))}) | {
        (cn, None) for cn in cns if cn != c
    }

    return frozenset({("r", attach), ("r", rest)})


def labeld_r_path(length: int, cns: set[str]) -> Concept:
    attach = frozenset({(cn, None) for cn in cns})
    if length <= 1:
        return attach
    else:
        return frozenset({("r", labeld_r_path(length - 1, cns))}) | attach


def is_core(c: Concept) -> bool:
    # If frontier element not weaker, then it is not a core
    for d in frontier(c):
        if subsum(c, d):
            return False
    return True


def remove_random_atom(c: Concept) -> Concept:
    A = concept2structure(c)

    atoms = []
    for cn in A[1].keys():
        for i in range(len(A[1][cn])):
            atoms.append(cn)
    for a in A[2].keys():
        for i in range(len(A[2][a])):
            atoms.append(a)

    if len(atoms) == 0:
        # Removal not possible
        return c

    atom = random.choice(atoms)
    if atom in A[1].keys():
        elem = random.choice(list(A[1][atom]))
        A[1][atom].remove(elem)
    elif atom in A[2].keys():
        elem = random.choice(list(A[2][atom]))
        A[2][atom].remove(elem)

    # This should ignore disconnected parts
    return structure2concept(A)


def execute_sml_bench(path, task):
    time_start = time.process_time()

    tasks = load_sml_tasks(path, task)

    time_parsed = time.process_time()
    for (lpname, (owlfile, A, P, N, indmap)) in tasks.items():
        print("== Starting incremental solving of {} {}".format(task, lpname))
        time_start_solve = time.process_time()

        res = solve_incr(A, P, N, mode.exact)

        time_solved = time.process_time()

        print(
            "== Took {:.2f}s for reading input and {:.3f}s for solving".format(
                time_parsed - time_start, time_solved - time_start_solve
            )
        )

        # verify_solution(owlfile, P, N, indmap, res)

        print()
