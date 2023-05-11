# SPELL

**SPELL** (SAT-bases PAC EL concept Learner) is an implementation of a
sample-efficient learning algorithm for EL-concepts under ELHr-ontologies.
That is, it takes as input a knowledge base containing an ontology written in
the description logic ELHr and lists of individuals from the knowledge base that
are positive or negative examples. It then outputs (if it exists) an EL-concept
that fits the examples under the ontology.

More information on SPELL and the theory behind it is available in the following paper:
*insert link to paper here*

You can find information on how to reproduce the benchmarks in [benchmarks.md](benchmarks.md)

Contact: Maurice Funk `mfunk@informatik.uni-leipzig.de`

## Setting up and running SPELL

These instructions were tests with python 3.10.0 on macOS.

Create a python virtual enviroment (in order to not install dependencies in the global environment):
```
    python -m venv spell-venv
```
Enter the virtual environment:
```
    source ./spell-venv/bin/activate
```
Install dependencies:
```
    pip install -r requirements.txt
```
Check that everything works by running the tests:
```
    pytest
```
Run an example:
```
    python spell_cli.py tests/father.owl tests/father-example/P.txt tests/father-example/N.txt
```
See
```
    python spell_cli.py --help
```
for some options.

