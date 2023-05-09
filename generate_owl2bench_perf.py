import sys

from spell.benchmark_tools import construct_sml_benchmark, parse_concept


def generate_owl2bench_benchmarks(sml_bench_path, owlfile):
    c1 = parse_concept(
        "<http://benchmark/OWL2Bench#UGCourse> \\sqcap \\exists <http://benchmark/OWL2Bench#isTaughtBy> ( <http://benchmark/OWL2Bench#Man> \\sqcap \\exists <http://benchmark/OWL2Bench#likes> <http://benchmark/OWL2Bench#Music> )"
    )

    construct_sml_benchmark(
        sml_bench_path, "owl2bench-1", owlfile, c1, weaken_steps=2, size_bound=100
    )

    c2 = parse_concept(
        "<http://benchmark/OWL2Bench#UGCourse> \\sqcap \\exists <http://benchmark/OWL2Bench#isTaughtBy> ( <http://benchmark/OWL2Bench#Woman> \\sqcap \\exists <http://benchmark/OWL2Bench#hasSameHomeTownWith> <http://benchmark/OWL2Bench#Student> )"
    )

    construct_sml_benchmark(
        sml_bench_path, "owl2bench-2", owlfile, c2, weaken_steps=3, size_bound=200
    )

    c3 = parse_concept(
        "<http://benchmark/OWL2Bench#UGCourse> \\sqcap \\exists <http://benchmark/OWL2Bench#isTaughtBy> ( <http://benchmark/OWL2Bench#Woman> \\sqcap \\exists <http://benchmark/OWL2Bench#isAssistantProfessorOf> \\top \\sqcap \\exists <http://benchmark/OWL2Bench#isCrazyAbout> \\top )"
    )

    construct_sml_benchmark(
        sml_bench_path, "owl2bench-3", owlfile, c3, weaken_steps=4, size_bound=200
    )

    c4 = parse_concept(
        "<http://benchmark/OWL2Bench#Woman> \\sqcap \\exists <http://benchmark/OWL2Bench#teachesCourse> <http://benchmark/OWL2Bench#UGCourse>"
    )
    construct_sml_benchmark(
        sml_bench_path,
        "owl2bench-4",
        owlfile,
        c4,
        weaken_steps=2,
        size_bound=100,
    )

    c5 = parse_concept(
        "<http://benchmark/OWL2Bench#Woman> \\sqcap \\exists <http://benchmark/OWL2Bench#teachesCourse> <http://benchmark/OWL2Bench#UGCourse> \\sqcap \\exists <http://benchmark/OWL2Bench#dislikes> \\top"
    )
    construct_sml_benchmark(
        sml_bench_path,
        "owl2bench-5",
        owlfile,
        c5,
        weaken_steps=2,
        size_bound=100,
    )

    c6 = parse_concept(
        "<http://benchmark/OWL2Bench#Woman> \\sqcap \\exists <http://benchmark/OWL2Bench#teachesCourse> <http://benchmark/OWL2Bench#UGCourse> \\sqcap \\exists <http://benchmark/OWL2Bench#dislikes> \\top \\sqcap \\exists <http://benchmark/OWL2Bench#isAssistantProfessorOf> \\top"
    )
    construct_sml_benchmark(
        sml_bench_path,
        "owl2bench-6",
        owlfile,
        c6,
        weaken_steps=2,
        size_bound=100,
    )


def main():
    if len(sys.argv) < 3:
        print(
            "Requires arguments: path-to-sml-bench-learningtasks path-to-owl2bench-owlfile"
        )
        return

    owlfile = "tests/OWL2EL-1.owl"
    owlfile = sys.argv[2]
    sml_bench_path = sys.argv[1]

    generate_owl2bench_benchmarks(sml_bench_path, owlfile)


if __name__ == "__main__":
    main()
