import sys

from spell.benchmark_tools import (Concept, concept2string,
                                   construct_owl_from_concepts,
                                   emit_sml_benchmark, frontier,
                                   remove_random_atom)


def construct_hard_conjunction_benchmark(size) -> tuple[list[Concept], list[Concept]]:
    C: Concept = frozenset(
        {("r", frozenset({("A{}".format(i), None) for i in range(size)}))}
    )

    return [C], frontier(C)


def construct_hard_deep_conjunction_benchmark(
    size,
) -> tuple[list[Concept], list[Concept]]:

    C: Concept = frozenset(
        {
            (
                "r",
                frozenset(
                    {("r", frozenset({("A{}".format(i), None) for i in range(size)}))}
                ),
            )
        }
    )

    return [C], frontier(C)


def construct_hard_path_benchmark(size) -> tuple[list[Concept], list[Concept]]:
    C = frozenset()

    for i in range(size):
        C = frozenset({("r", C)})

    return [C], frontier(C)


def main():
    if len(sys.argv) < 2:
        print("Requires argument: path to sml-bench learningtasks")
        return

    sml_bench_path = sys.argv[1]

    for size in range(1, 19):
        P, N = construct_hard_deep_conjunction_benchmark(size)

        Ps, Ns = construct_owl_from_concepts("temp.owl", P, N)

        emit_sml_benchmark(
            sml_bench_path,
            "test-hard-deep-conj-{}".format(size),
            "temp.owl",
            Ps,
            Ns,
            ["Target query: {}".format(concept2string(P[0]))],
        )

    for size in range(1, 19):
        P, N = construct_hard_conjunction_benchmark(size)

        Ps, Ns = construct_owl_from_concepts("temp.owl", P, N)

        emit_sml_benchmark(
            sml_bench_path,
            "test-hard-conj-{}".format(size),
            "tmp.owl",
            Ps,
            Ns,
            ["Target query: {}".format(concept2string(P[0]))],
        )
    for size in range(1, 19):
        P, N = construct_hard_path_benchmark(size)

        Ps, Ns = construct_owl_from_concepts("temp.owl", P, N)

        emit_sml_benchmark(
            sml_bench_path,
            "test-hard-path-{}".format(size),
            "tmp.owl",
            Ps,
            Ns,
            ["Target query: {}".format(concept2string(P[0]))],
        )
    return


if __name__ == "__main__":
    main()
