from dataclasses import dataclass
from flask import Flask, render_template
from flask import Flask, render_template, request, redirect, url_for

from spell.fitting import solve_incr, mode, solve_incr2

from spell.structures import solution2sparql, structure_from_owl

@dataclass
class Example:
    id: int = 0
    name: str = ""

state: list[Example] = []

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("base.html", todo_list=state)

@app.route("/run", methods = ["POST"])
def generate_output():
    data = request.get_json()
    def generate():
        P = data["P"]
        N = data["N"]

        _, res = yield from solve_incr2(A, P, N, mode[data["mode"]], max_size=8)
    return app.response_class(generate(), mimetype='text/text')

indmap = {}


def load_structure():
    global indmap, A
    # A, indmap, _ = structure_from_owl("tests/family-benchmark.owl")
    A, indmap, _ = structure_from_owl("./yago-full.owl")
    limit = 15000

    # for k in indmap.keys():
    #     if limit <= 0:
    #         continue
    #     if not "#" in k:
    #         continue
    #     limit -= 1
    #     state.append(Example(indmap[k], k))

    reverse_indmap = {
        n: name
        for (name, n) in indmap.items()
        if "#" in name or "/" in name or "NC_" in name
    }
    for cn in A[1].keys():
        if "Movie" in cn:
            for id in A[1][cn]:
                if limit <=0:
                    break
                limit -= 1
                state.append(Example(id, reverse_indmap[id]))

    state.sort(key = lambda k : k.name)

if __name__ == "__main__":
    load_structure()
    app.run(debug=True)
