from dataclasses import dataclass
from flask import Flask, render_template, request

from spell.fitting import solve_incr, mode

from spell.structures import Structure, structure_from_owl

@dataclass
class Example:
    id: int = 0
    name: str = ""



import io
import contextlib

import functools
print = functools.partial(print, flush=True)

# Search for a small separating query by incrementally increasing the size
def solve_incr_generator(
    A: Structure,
    P: list[int],
    N: list[int],
    m: mode,
    timeout: float = -1,
    max_size: int = 19,
) :
    
    # Create a StringIO object to capture stdout
    f = io.StringIO()

    # Redirect stdout to the StringIO object
    with contextlib.redirect_stdout(f):
        solve_incr(A, P, N, m, timeout, max_size)

    f.seek(0)  
    for line in f.readlines():
        yield line



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
        yield from solve_incr_generator(A, P, N, mode[data["mode"]], max_size=8)
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
    A, indmap, _ = structure_from_owl("yago-tiny.owl")
    # A, indmap, _ = structure_from_owl("./yago-full.owl")


if __name__ == "__main__":
    load_structure()
    app.run(debug=True)
