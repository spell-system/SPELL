import random
import sys

from spell.benchmark_tools import (concept2sparql, concept2string,
                                   create_materialized_tdb_dir,
                                   create_restricted_owl, emit_sml_benchmark,
                                   get_reachable_inds, owlname2tdbname,
                                   parse_concept, query_tdbdir)


def generate_yago_generalization_bench(sml_bench_dir, owlfile):

    c4 = parse_concept(
        "\\exists <http://schema.org/actor> ( \\exists <http://schema.org/alumniOf> \\top \\sqcap \\exists <http://schema.org/award> \\top \\sqcap \\exists <http://schema.org/children> \\top \\sqcap \\exists <http://schema.org/deathPlace> \\top )"
    )
    nc = parse_concept("\\exists <http://schema.org/actor> \\top")

    owlfile = "robot/yago-reasoned.owl"

    tdbdir = owlname2tdbname(owlfile)

    create_materialized_tdb_dir(owlfile, tdbdir)

    P = set(query_tdbdir(tdbdir, concept2sparql(c4)))

    N = set(query_tdbdir(tdbdir, concept2sparql(nc))) - P

    total = P | N
    totall = list(total)

    for nex in range(5, 80, 5):
        Ps = []
        Ns = []

        starts = set()
        for iopp in range(20):
            Pl = []
            Nl = []

            while (
                len(Pl) == 0
            ):  # At least one positive example, othwerwise ELTL complains
                Pl.clear()
                Nl.clear()
                for i in range(nex):
                    ind = random.choice(totall)
                    if ind in P:
                        Pl.append(ind)
                    if ind in N:
                        Nl.append(ind)

            starts |= set(Pl) | set(Nl)
            Ps.append(Pl)
            Ns.append(Nl)

        tmp_owl = "tmp.owl"

        print("== Collecting reachable individuals for this benchmark")
        relevant_inds = get_reachable_inds(owlfile, list(starts))

        rinds = []
        for ind in relevant_inds:
            # Remove things that eltl has trouble with
            if (
                ".png" in ind
                or ".svg" in ind
                or ".jpg" in ind
                or ".jpeg" in ind
                or ".JPG" in ind
                or "geo.com" in ind
            ):
                continue
            rinds.append(ind)

        print("== Creating reachable fragment of {}".format(owlfile))
        create_restricted_owl(owlfile, rinds, tmp_owl)

        for i in range(len(Ps)):
            emit_sml_benchmark(
                sml_bench_dir,
                "yago-gen-test-{}-{}".format(nex, i),
                tmp_owl,
                Ps[i],
                Ns[i],
                ["Target query: {}".format(concept2string(c4))],
            )


def main():
    if len(sys.argv) < 3:
        print(
            "Requires arguments: path-to-sml-bench-learningtasks path-to-yago-owlfile"
        )
        return

    owlfile = "robot/yago-reasoned.owl"
    owlfile = sys.argv[2]
    sml_bench_path = sys.argv[1]

    generate_yago_generalization_bench(sml_bench_path, owlfile)


if __name__ == "__main__":
    main()
