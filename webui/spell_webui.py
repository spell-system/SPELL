from dataclasses import dataclass
from flask import Flask, render_template
from flask import Flask, render_template, request, redirect, url_for

from spell.fitting import solve_incr, mode, solve_incr2

from spell.structures import solution2sparql, structure_from_owl

@dataclass
class Example:
    id: int = 0
    name: str = ""

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("base.html")

@app.route("/run", methods = ["POST"])
def generate_output():
    data = request.get_json()
    def generate():
        P = data["P"]
        N = data["N"]

        _, res = yield from solve_incr2(A, P, N, mode[data["mode"]], max_size=8)
    return app.response_class(generate(), mimetype='text/text')

indmap = {}

@app.route("/search", methods = ["GET"])
def search():
    query = request.args.get('q', '')

    result = []
    limit = 100
    for k in indmap:
        if limit <= 0:
            break
        
        if query in k.lower():
            result.append(Example(indmap[k], to_readable_name(k)))
            limit -= 1

    result.sort(key = lambda k : k.name)

    return result




def to_readable_name(name) -> str:
    return name


def load_structure():
    global indmap, A
    # A, indmap, _ = structure_from_owl("tests/fm.owl")
    A, indmap, _ = structure_from_owl("./yago-full.owl")


if __name__ == "__main__":
    load_structure()
    app.run(debug=True)
