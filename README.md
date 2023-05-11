# SPELL

**SPELL** (SAT-bases PAC EL concept Learner) is an implementation of a
sample-efficient learning algorithm for EL-concepts under ELHr-ontologies.
It takes as input a knowledge base (formulated in the description logic ELHr)
and lists of individuals from that knowledge base that
are positive or negative examples. 
It then *learns* an EL-concept that fits the examples with respect to the
provided background knowledge.

More information on SPELL and the theory behind it is available in the following paper:
*insert link to paper here*

You can find instructions on how to reproduce the benchmarks in [benchmarks.md](benchmarks.md)

Contact: Maurice Funk `mfunk@informatik.uni-leipzig.de`

## Setting up and running SPELL

These instructions were tested with python 3.10.9 on macOS.

Create a python virtual enviroment (to avoid installing dependencies in the global environment):
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
Make sure that the `robot` tool is available in the `robot` directory (this is required for some tests):
```
cd robot
./get_robot.sh
cd ..
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

