import time

from spell.fitting import solve_incr, solve, mode
from spell.structures import ABoxBuilder, compact_canonical_model, structure_from_owl, Structure, structure_to_dot
from spell.benchmark_tools import execute_sml_bench


def solve_k2(size, A, P, N):
    sol = solve(size, A, P, N, 1, True)
    if sol is None:
        return False
    cov, q = sol
    return cov == len(P) + len(N)


def solve_incr2(A, P, N) -> bool:
    cov, q = solve_incr(A, P, N, mode.neg_approx)
    return cov == len(P) + len(N)


def test_cycle():
    A, indmap, _ = structure_from_owl("tests/test-cycle.owl")

    P = [
        indmap["test.p1"],
    ]
    N = [indmap["test.n1"]]

    assert solve_incr2(A, P, N)


def test_hard():
    A, indmap, _ = structure_from_owl("tests/test-hard-11.owl")

    P = [
        indmap["test.p"],
    ]
    N = [indmap["test.n"]]

    time_start = time.process_time()
    assert solve_incr2(A, P, N)
    time_end = time.process_time()
    print("{}".format(time_end - time_start))



def test_father():
    A, indmap, _ = structure_from_owl("tests/father.owl")

    P = [indmap["http://example.com/father#markus"], indmap["http://example.com/father#stefan"], indmap["http://example.com/father#martin"]]
    N = [indmap["http://example.com/father#heinz"], indmap["http://example.com/father#anna"], indmap["http://example.com/father#michelle"]]
    assert not solve_k2(1, A, P, N)
    assert solve_k2(2, A, P, N)
    assert solve_incr2(A, P, N)


def test_fm_bench():
    A, indmap, _ = structure_from_owl("tests/family-benchmark.owl")

    P = [
        indmap[i]
        for i in [
            "family.F2M18",
            "family.F2M11",
            "family.F2M25",
            "family.F2M23",
            "family.F2M21",
            "family.F2M32",
            "family.F2M35",
            "family.F3M44",
            "family.F3M51",
            "family.F3M47",
            "family.F3M45",
            "family.F5M68",
            "family.F5M66",
            "family.F5M64",
            "family.F6M75",
            "family.F6M73",
            "family.F6M71",
            "family.F6M81",
            "family.F6M90",
            "family.F6M99",
            "family.F6M100",
            "family.F6M92",
            "family.F7M112",
            "family.F7M110",
            "family.F7M113",
            "family.F7M117",
            "family.F7M115",
            "family.F7M125",
            "family.F7M123",
            "family.F7M131",
            "family.F7M104",
            "family.F8M138",
            "family.F8M136",
            "family.F8M134",
            "family.F9M147",
            "family.F9M151",
            "family.F9M155",
            "family.F9M153",
            "family.F9M161",
            "family.F9M159",
            "family.F9M166",
            "family.F9M162",
            "family.F9M157",
            "family.F9M167",
            "family.F10M173",
            "family.F10M183",
            "family.F10M184",
            "family.F10M190",
            "family.F10M188",
            "family.F10M199",
            "family.F10M197",
        ]
    ]

    N = [
        indmap[i]
        for i in [
            "family.F6M78",
            "family.F9F152",
            "family.F9M149",
            "family.F2F19",
            "family.F9F141",
            "family.F10M202",
            "family.F7M109",
            "family.F6F84",
            "family.F7F105",
            "family.F10M187",
            "family.F9F169",
            "family.F2M20",
            "family.F1F7",
            "family.F8F133",
            "family.F2F24",
            "family.F6F83",
            "family.F10F186",
            "family.F9F164",
            "family.F9M144",
            "family.F1F2",
            "family.F7F121",
            "family.F8F135",
            "family.F6F94",
            "family.F6F74",
            "family.F9F150",
            "family.F2F22",
            "family.F5F65",
            "family.F6F79",
            "family.F6F82",
            "family.F9M139",
            "family.F6M88",
            "family.F9F160",
            "family.F6F91",
            "family.F6F93",
            "family.F7F108",
            "family.F4F58",
            "family.F2F15",
            "family.F6M95",
            "family.F9F163",
            "family.F10F175",
            "family.F1M4",
            "family.F9M142",
            "family.F3F41",
            "family.F9F143",
            "family.F8F137",
            "family.F10M171",
            "family.F3F42",
            "family.F7F111",
            "family.F1F3",
            "family.F7M102",
            "family.F3M43",
            "family.F1M6",
        ]
    ]
    assert solve_incr2(A, P, N)


# test_fm_bench()


def test_fm_bench2():
    A, indmap, _ = structure_from_owl("tests/family-benchmark.owl")

    P = [
        indmap[i]
        for i in [
            "family.F2M13",
            "family.F2M18",
            "family.F2M25",
            "family.F2M23",
            "family.F2M21",
            "family.F2M32",
            "family.F2M35",
            "family.F3M44",
            "family.F3M51",
            "family.F3M47",
            "family.F3M45",
            "family.F5M68",
            "family.F5M66",
            "family.F6M75",
            "family.F6M73",
            "family.F6M81",
            "family.F6M90",
            "family.F6M99",
            "family.F6M100",
            "family.F7M112",
            "family.F7M110",
            "family.F7M113",
            "family.F7M117",
            "family.F7M115",
            "family.F7M125",
            "family.F7M123",
            "family.F7M131",
            "family.F8M138",
            "family.F8M136",
            "family.F9M147",
            "family.F9M151",
            "family.F9M155",
            "family.F9M153",
            "family.F9M161",
            "family.F9M159",
            "family.F9M166",
            "family.F9M162",
            "family.F10M183",
            "family.F10M184",
            "family.F10M190",
            "family.F10M188",
            "family.F10M199",
            "family.F10M197",
        ]
    ]

    N = [
        indmap[i]
        for i in [
            "family.F6F83",
            "family.F4M57",
            "family.F1M8",
            "family.F9F140",
            "family.F4F58",
            "family.F2M29",
            "family.F9M170",
            "family.F7F118",
            "family.F2F19",
            "family.F2M16",
            "family.F2M34",
            "family.F10M182",
            "family.F7M120",
            "family.F8M134",
            "family.F6F74",
            "family.F10F192",
            "family.F6F86",
            "family.F2F28",
            "family.F9M139",
            "family.F10M194",
            "family.F1F5",
            "family.F4F56",
            "family.F6F89",
            "family.F2F33",
            "family.F10F174",
            "family.F7M128",
            "family.F7F129",
            "family.F9F158",
            "family.F3M50",
            "family.F6F94",
            "family.F7F114",
            "family.F6F72",
            "family.F7F124",
            "family.F9F150",
            "family.F4F55",
            "family.F10F175",
            "family.F1F7",
            "family.F4M54",
            "family.F7F108",
            "family.F6M92",
            "family.F9F152",
            "family.F6M85",
            "family.F2F36",
        ]
    ]

    assert not solve_k2(1, A, P, N)
    assert not solve_k2(2, A, P, N)
    assert solve_k2(3, A, P, N)
    assert solve_incr2(A, P, N)


def test_carcinogen():
    A, indmap, _ = structure_from_owl("tests/carcinogenesis.owl")

    P = [
        indmap[i]
        for i in [
            "carcinogenesis.d1",
            "carcinogenesis.d10",
            "carcinogenesis.d101",
            "carcinogenesis.d102",
            "carcinogenesis.d103",
            "carcinogenesis.d106",
            "carcinogenesis.d107",
            "carcinogenesis.d108",
            "carcinogenesis.d11",
            "carcinogenesis.d12",
            "carcinogenesis.d13",
            "carcinogenesis.d134",
            "carcinogenesis.d135",
            "carcinogenesis.d136",
            "carcinogenesis.d138",
            "carcinogenesis.d140",
            "carcinogenesis.d141",
            "carcinogenesis.d144",
            "carcinogenesis.d145",
            "carcinogenesis.d146",
            "carcinogenesis.d147",
            "carcinogenesis.d15",
            "carcinogenesis.d17",
            "carcinogenesis.d19",
            "carcinogenesis.d192",
            "carcinogenesis.d193",
            "carcinogenesis.d195",
            "carcinogenesis.d196",
            "carcinogenesis.d197",
            "carcinogenesis.d198",
            "carcinogenesis.d199",
            "carcinogenesis.d2",
            "carcinogenesis.d20",
            "carcinogenesis.d200",
            "carcinogenesis.d201",
            "carcinogenesis.d202",
        ]
    ]

    N = [
        indmap[i]
        for i in [
            "carcinogenesis.d110",
            "carcinogenesis.d111",
            "carcinogenesis.d114",
            "carcinogenesis.d116",
            "carcinogenesis.d117",
            "carcinogenesis.d119",
            "carcinogenesis.d121",
            "carcinogenesis.d123",
            "carcinogenesis.d124",
            "carcinogenesis.d125",
            "carcinogenesis.d127",
            "carcinogenesis.d128",
            "carcinogenesis.d130",
            "carcinogenesis.d133",
            "carcinogenesis.d150",
            "carcinogenesis.d151",
            "carcinogenesis.d154",
            "carcinogenesis.d155",
            "carcinogenesis.d156",
            "carcinogenesis.d159",
            "carcinogenesis.d160",
            "carcinogenesis.d161",
            "carcinogenesis.d162",
            "carcinogenesis.d163",
            "carcinogenesis.d164",
            "carcinogenesis.d165",
            "carcinogenesis.d166",
            "carcinogenesis.d169",
            "carcinogenesis.d170",
            "carcinogenesis.d171",
            "carcinogenesis.d172",
            "carcinogenesis.d173",
            "carcinogenesis.d174",
            "carcinogenesis.d178",
            "carcinogenesis.d179",
            "carcinogenesis.d180",
            "carcinogenesis.d181",
            "carcinogenesis.d183",
            "carcinogenesis.d184",
            "carcinogenesis.d185",
            "carcinogenesis.d186",
            "carcinogenesis.d188",
            "carcinogenesis.d190",
            "carcinogenesis.d194",
            "carcinogenesis.d207",
            "carcinogenesis.d208_1",
        ]
    ]

    assert not solve_k2(1, A, P, N)
    assert not solve_k2(2, A, P, N)


# test_carcinogen()


def test_lymphography():
    A, indmap, _ = structure_from_owl("tests/lymphography.owl")

    P = [
        indmap[i]
        for i in [
            "lymphography.2",
            # "lymphography.5",
            # "lymphography.6",
            # "lymphography.7",
            # "lymphography.8",
            "lymphography.11",
            # "lymphography.12",
            # "lymphography.13",
            # "lymphography.16",
            # "lymphography.17",
            # "lymphography.18",
            # "lymphography.19",
            # "lymphography.20",
            # "lymphography.23",
            # "lymphography.24",
            # "lymphography.29",
            # "lymphography.33",
            # "lymphography.34",
            # "lymphography.35",
            # "lymphography.36",
            # "lymphography.43",
            # "lymphography.44",
            # "lymphography.47",
            # "lymphography.48",
            # "lymphography.49",
            # "lymphography.50",
            # "lymphography.51",
            # "lymphography.53",
            # "lymphography.54",
            # "lymphography.55",
            # "lymphography.58",
            # "lymphography.60",
            # "lymphography.62",
            # "lymphography.64",
            # "lymphography.65",
            # "lymphography.68",
            # "lymphography.71",
            # "lymphography.73",
            # "lymphography.76",
            # "lymphography.77",
            # "lymphography.78",
            # "lymphography.82",
            # "lymphography.86",
            # "lymphography.87",
            # "lymphography.88",
            # "lymphography.90",
            # "lymphography.92",
            # "lymphography.93",
            # "lymphography.94",
            # "lymphography.97",
            # "lymphography.98",
            # "lymphography.99",
            # "lymphography.100",
            # "lymphography.102",
            # "lymphography.103",
            # "lymphography.104",
            # "lymphography.105",
            # "lymphography.107",
            # "lymphography.109",
            # "lymphography.110",
            # "lymphography.111",
            # "lymphography.113",
            # "lymphography.118",
            # "lymphography.121",
            # "lymphography.122",
            # "lymphography.123",
            # "lymphography.124",
            # "lymphography.125",
            # "lymphography.127",
            # "lymphography.128",
            # "lymphography.130",
            # "lymphography.131",
            # "lymphography.134",
            # "lymphography.135",
            # "lymphography.136",
            # "lymphography.139",
            # "lymphography.141",
            # "lymphography.143",
            # "lymphography.145",
            # "lymphography.147",
            # "lymphography.148"
        ]
    ]

    N = [
        indmap[i]
        for i in [
            "lymphography.79",
            "lymphography.137",
            "lymphography.1",
            "lymphography.3",
            "lymphography.4",
            "lymphography.9",
            "lymphography.10",
            "lymphography.14",
            "lymphography.21",
            "lymphography.22",
            "lymphography.25",
            "lymphography.26",
            "lymphography.27",
            "lymphography.28",
            "lymphography.30",
            "lymphography.31",
            "lymphography.32",
            "lymphography.38",
            "lymphography.39",
            "lymphography.40",
            "lymphography.41",
            "lymphography.42",
            "lymphography.46",
            "lymphography.52",
            "lymphography.56",
            "lymphography.57",
            "lymphography.59",
            "lymphography.61",
            "lymphography.63",
            "lymphography.66",
            "lymphography.67",
            "lymphography.69",
            "lymphography.70",
            "lymphography.72",
            "lymphography.74",
            "lymphography.75",
            "lymphography.80",
            "lymphography.81",
            "lymphography.83",
            "lymphography.84",
            "lymphography.85",
            "lymphography.89",
            "lymphography.91",
            "lymphography.95",
            "lymphography.96",
            "lymphography.101",
            "lymphography.106",
            "lymphography.108",
            "lymphography.112",
            "lymphography.114",
            "lymphography.115",
            "lymphography.116",
            "lymphography.117",
            "lymphography.119",
            "lymphography.120",
            "lymphography.126",
            "lymphography.129",
            "lymphography.132",
            "lymphography.133",
            "lymphography.138",
            "lymphography.142",
            "lymphography.144",
            "lymphography.146",
            "lymphography.15",
            "lymphography.37",
            "lymphography.45",
            "lymphography.140",
        ]
    ]

    assert solve_incr2(A, P, N)


def test_trains():
    A, indmap, _ = structure_from_owl("tests/trains.owl")

    P = [
        indmap[i]
        for i in [
            "trains.east1",
            "trains.east2",
            "trains.east3",
            "trains.east4",
            "trains.east5",
        ]
    ]

    N = [
        indmap[i]
        for i in [
            "trains.west6",
            "trains.west7",
            "trains.west8",
            "trains.west9",
            "trains.west10",
        ]
    ]

    assert solve_incr2(A, P, N)


def test_owl1():
    execute_sml_bench("tests/", "owl2bench-1")


def test_owlbench():
    execute_sml_bench("tests/", "owl2bench-4")


def test_el_path5():
    A, i, _ = structure_from_owl("tests/test-el-path-5.owl")

    P = [
        i["http://example.com/test#p1"],
        i["http://example.com/test#p2"],
    ]

    N = [
        i["http://example.com/test#n"],
    ]

    assert not solve_k2(6, A, P, N)
    assert solve_k2(7, A, P, N)


def test_el_path4():
    A, i, _ = structure_from_owl("tests/test-el-path-4.owl")

    P = [
        i["http://example.com/test#p1"],
        i["http://example.com/test#p2"],
    ]

    N = [
        i["http://example.com/test#n"],
    ]

    assert solve_incr2(A, P, N)
    assert not solve_k2(5, A, P, N)
    assert solve_k2(6, A, P, N)


def test_el_tree15():
    A, i, _ = structure_from_owl("tests/test-el-tree-15.owl")

    P = [
        i["http://example.com/test#p"],
    ]

    N = [
        i["http://example.com/test#n"],
    ]

    # There should be no separating concept, if tbox reasoning works
    assert not solve_k2(5, A, P, N)


def test_owl_bench():
    A, i, _ = structure_from_owl("tests/OWL2EL-1.owl")

    P = [i["http://benchmark/OWL2Bench#U0WC0D3UGWS9"]]

    N = [
        i["http://benchmark/OWL2Bench#U0WC0D3VP0"],
    ]

    assert solve_incr2(A, P, N)


# test_owl_bench()

def test_resoning1():
    from spell.structures import TBox

    t = TBox("top")
    for i in range(1, 15):
        t.add_axiom3("A{}".format(i), "r{}".format(i), "top")
        t.add_range_restriction("r{}".format(i), "A{}".format(i + 1))
        t.add_role_inc("r{}".format(i), "r")

    t.add_axiom3("A", "s", "A")
    t.add_role_inc("s", "r")
    t.add_axiom3("B", "t", "B")
    t.add_role_inc("t", "r")

    Ab = ABoxBuilder()
    Ab.concept_assertion(Ab.map_ind("a"), "A")
    Ab.concept_assertion(Ab.map_ind("a"), "A1")
    Ab.concept_assertion(Ab.map_ind("b"), "A1")
    Ab.concept_assertion(Ab.map_ind("c"), "B")

    t.saturate()
    compact_canonical_model(Ab, t)

    assert solve_incr2(Ab.A, [Ab.indmap["a"], Ab.indmap["c"]], [Ab.indmap["b"]])

def test_resoning2():
    from spell.structures import TBox

    t = TBox("top")
    t.add_axiom4("r", "A", "A")
    t.add_axiom3("A", "s", "A")
    t.add_role_inc("r", "s")
    t.add_role_inc("s", "r")


    Ab = ABoxBuilder()
    Ab.declare_rn("s")
    for i in range(1, 15):
        Ab.role_assertion(Ab.map_ind("a{}".format(i)), "a{}".format(i + 1), "s")

    Ab.concept_assertion(Ab.map_ind("a14"), "A")
    Ab.concept_assertion(Ab.map_ind("b"), "A")
    
    t.saturate()
    compact_canonical_model(Ab, t)

    res = solve_incr(Ab.A, [Ab.indmap["a1"]], [Ab.indmap["b"]], mode.exact, max_size=10)

    assert res[0] == 1

# test_resoning()

def test_TBox():
    from spell.structures import TBox

    t = TBox("top")

    t.add_axiom1("0", "A")
    t.add_axiom3("A", "r", "B")
    t.add_role_inc("r", "s")
    t.add_range_restriction("s", "C")
    t.add_axiom2("B", "C", "D")
    t.add_axiom4("r", "D", "E")
    t.add_axiom1("E", "1")

    t.saturate()

    assert "1" in t.implic["0"]
