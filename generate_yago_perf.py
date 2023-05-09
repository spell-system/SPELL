import sys

from spell.benchmark_tools import (generate_benchmark_collection,
                                   parse_concept, weaken_drop_root_subtrees)


def generate_yago_perf(sml_bench_dir, owlfile):
    c10 = parse_concept(
        "\\exists <http://schema.org/actor> ( \\exists <http://schema.org/alumniOf> \\top \\sqcap \\exists <http://schema.org/award> \\top \\sqcap \\exists <http://schema.org/children> \\top \\sqcap \\exists <http://schema.org/deathPlace> \\top \\sqcap \\exists <http://schema.org/parent> \\top \\sqcap \\exists <http://schema.org/knowsLanguage> \\top \\sqcap \\exists <http://schema.org/spouse> \\top \\sqcap \\exists <http://schema.org/birthPlace> \\top \\sqcap \\exists <http://schema.org/familyName> \\top \\sqcap \\exists <http://schema.org/givenName> \\top )"
    )
    c9 = parse_concept(
        "\\exists <http://schema.org/actor> ( \\exists <http://schema.org/alumniOf> \\top \\sqcap \\exists <http://schema.org/award> \\top \\sqcap \\exists <http://schema.org/children> \\top \\sqcap \\exists <http://schema.org/deathPlace> \\top \\sqcap \\exists <http://schema.org/parent> \\top \\sqcap \\exists <http://schema.org/knowsLanguage> \\top \\sqcap \\exists <http://schema.org/spouse> \\top \\sqcap \\exists <http://schema.org/birthPlace> \\top \\sqcap \\exists <http://schema.org/familyName> \\top )"
    )
    c8 = parse_concept(
        "\\exists <http://schema.org/actor> ( \\exists <http://schema.org/alumniOf> \\top \\sqcap \\exists <http://schema.org/award> \\top \\sqcap \\exists <http://schema.org/children> \\top \\sqcap \\exists <http://schema.org/deathPlace> \\top \\sqcap \\exists <http://schema.org/parent> \\top \\sqcap \\exists <http://schema.org/knowsLanguage> \\top \\sqcap \\exists <http://schema.org/spouse> \\top \\sqcap \\exists <http://schema.org/birthPlace> \\top )"
    )

    c7 = parse_concept(
        "\\exists <http://schema.org/actor> ( \\exists <http://schema.org/alumniOf> \\top \\sqcap \\exists <http://schema.org/award> \\top \\sqcap \\exists <http://schema.org/children> \\top \\sqcap \\exists <http://schema.org/deathPlace> \\top \\sqcap \\exists <http://schema.org/parent> \\top \\sqcap \\exists <http://schema.org/knowsLanguage> \\top \\sqcap \\exists <http://schema.org/spouse> \\top )"
    )

    c6 = parse_concept(
        "\\exists <http://schema.org/actor> ( \\exists <http://schema.org/alumniOf> \\top \\sqcap \\exists <http://schema.org/award> \\top \\sqcap \\exists <http://schema.org/children> \\top \\sqcap \\exists <http://schema.org/deathPlace> \\top \\sqcap \\exists <http://schema.org/parent> \\top \\sqcap \\exists <http://schema.org/knowsLanguage> \\top )"
    )

    c5 = parse_concept(
        "\\exists <http://schema.org/actor> ( \\exists <http://schema.org/alumniOf> \\top \\sqcap \\exists <http://schema.org/award> \\top \\sqcap \\exists <http://schema.org/children> \\top \\sqcap \\exists <http://schema.org/deathPlace> \\top \\sqcap \\exists <http://schema.org/parent> \\top )"
    )

    c4 = parse_concept(
        "\\exists <http://schema.org/actor> ( \\exists <http://schema.org/alumniOf> \\top \\sqcap \\exists <http://schema.org/award> \\top \\sqcap \\exists <http://schema.org/children> \\top \\sqcap \\exists <http://schema.org/deathPlace> \\top )"
    )

    c3 = parse_concept(
        "\\exists <http://schema.org/actor> ( \\exists <http://schema.org/alumniOf> \\top \\sqcap \\exists <http://schema.org/award> \\top \\sqcap \\exists <http://schema.org/children> \\top )"
    )
    c2 = parse_concept(
        "\\exists <http://schema.org/actor> ( \\exists <http://schema.org/alumniOf> \\top \\sqcap \\exists <http://schema.org/award> \\top )"
    )

    # owlfile = "robot/yago-reasoned.owl"
    cs = [c2, c3, c4, c5, c6, c7, c8, c9, c10]
    concepts = []

    for i in range(2, 11):
        ci = cs[i - 2]
        concepts.append(("{}".format(i), ci, weaken_drop_root_subtrees(ci, 1)))

    for exc in [20, 40, 60, 80, 100, 120]:
        generate_benchmark_collection(
            sml_bench_dir,
            "yago-1-succ-{}-ex-reachable".format(exc),
            owlfile,
            concepts,
            exc,
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

    generate_yago_perf(sml_bench_path, owlfile)


if __name__ == "__main__":
    main()
