# Reproducing the benchmarks

This directory includes
* SPELL (in the `spell` directory)
* Results of the experiments (in the `benchmarks/results/` directory)
* Configuration files for running the benchmarks with SML-Bench (in the `benchmarks/` directory)
* Benchmark generators (in form of the `generate_*.py` scripts)
* Integration of SPELL into SML-Bench (in the `sml_bench_integration` directory)

The benchmarks themselves are not included, due to size restrictions. They are
available separately at 
 https://github.com/spell-system/benchmarks, but can also be generated using the included benchmark generators.


Make sure that the `robot` tool is available in the `robot` directory:
```
cd robot
./get_robot.sh
cd ..
```

## Setting up SML-Bench Framework

The following instructions show how to set up the SML-Bench framework (https://github.com/SmartDataAnalytics/SML-Bench) in order to run the benchmarks.
Additionally there are instructions to set up another EL concept learning
system, DL-Learner. If you are just interested in running the benchmarks with
SPELL, you can skip all steps involving DL-Learner.

The instructions were tested with openjdk 19 and maven 3.8.7.

### Step 1: Build the SML-Bench Framework

Clone the SML-Bench repository
```
git clone https://github.com/SmartDataAnalytics/SML-Bench.git
cd SML-Bench
git checkout 278dbeab8224776bc97f77b71002f2eefe659589
```

Package SML-Bench and unpack it 
(without running the tests, as there seems to be some maven problem)
```
mvn package -DskipTests
cd ..
unzip SML-Bench/target/sml-bench-core-0.3-SNAPSHOT.zip
```

The directory `sml-bench-core-0.3-SNAPSHOT` should now exist and contain e.g. the directories `bin` and `learningsystems`.

### Step 2: Build DL-Learner

Clone the DL-Learner repository
```
git clone https://github.com/SmartDataAnalytics/DL-Learner.git
cd DL-Learner
git checkout a7cd4441e52b6e54aefdea33a4914e9132ebfd97 
```
Build DL-Learner
```
./buildRelease.sh
```

Unpack DL-Learner into the `sml-bench-core-0.3-SNAPSHOT/learningsystems/dllearner` directory
```
cd ../sml-bench-core-0.3-SNAPSHOT/learningsystems/dllearner
unzip ../../../DL-Learner/interfaces/target/dllearner-1.5.1-SNAPSHOT.zip
cd ../..
```


### Step 3: Make SML-Bench and DL-Learner work in 2023

From here on, the guide assumes that we are in the `sml-bench-core-0.3-SNAPSHOT/` directory.

On MacOs: you need GNU coreutils to run SML-Bench. If you use homebrew, you could use it to install GNU coreutils:
```
brew install coreutils grep gnu-sed
PATH="$(brew --prefix)/opt/coreutils/libexec/gnubin:$PATH"
PATH="$(brew --prefix)/opt/grep/libexec/gnubin:$PATH"
PATH="$(brew --prefix)/opt/gnu-sed/libexec/gnubin:$PATH"
```

DL-Learner does not work out of the box with recent JDK versions, it requires
the option `--add-opens java.base/java.lang=ALL-UNNAMED`.
Add the line
```
  --add-opens java.base/java.lang=ALL-UNNAMED \
```
to the `exec` command at the end of `learningsystems/dllearner/dllearner-1.5.1-SNAPSHOT/bin/cli`

And while you are editing the file, you might also want to increase the maximum heap size of DL-Learner:

Change `-Xmx2g` to `-Xmx10g`.

### Step 4: Configure DL-Learner to use ELTL
(unfortunately there is no better way, the problem specific config files cannot overwrite this setting)

Edit `learningsystems/dllearner/run`. Replace `celoe` on line 95 with `eltl`.

### Step 5: Setup SPELL as a SML-Bench learningsystem

Create the `learningsystems/spell` directory

Copy the files `run`, `validate` and `system.ini` from the
`../sml_bench_integration` directory to `learningsystems/spell`.

Create a symlink from `learningsystems/spell/spell` to the directoy `spell` of SPELL:
```
    ls -s PATH_TO_SPELL/spell learningsystems/spell
```

### Step 6: Check that everything works

Make sure that you are inside the SPELL virtualenv and if you are on MacOs, that
the GNU coreutils are in your `PATH`.

Create the file `test.plist` with the following content:
```
{
  config = 1;
  learningsystems = ( dllearner, spell);
  scenarios = ( animals/fish );
  measures = ( pred_acc );
  framework = {
    crossValidationFolds = 1;
    maxExecutionTime = 60;
  };
  resultOutput = testResult.prop;
}
```

Run
```
bin/smlbench test.plist
```

After the command finishes, there the results should be available in `testResult.prop`.
It should look like this:

```
animals.fish.absolute.dllearner.duration = 1
animals.fish.absolute.dllearner.trainingRaw = HasGills
animals.fish.absolute.dllearner.train_status = ok
animals.fish.absolute.dllearner.validationResult = ok
animals.fish.absolute.dllearner.ValidationRaw.tp = 4
animals.fish.absolute.dllearner.ValidationRaw.fp = 0
animals.fish.absolute.dllearner.ValidationRaw.tn = 11
animals.fish.absolute.dllearner.ValidationRaw.fn = 0
animals.fish.absolute.dllearner.measure.pred_acc = 1.0
animals.fish.absolute.spell.duration = 0
animals.fish.absolute.spell.trainingRaw = SELECT DISTINCT ?0 WHERE {, ?0 a <http://dl-learner.org/benchmark/dataset/animals/HasGills> .,}
animals.fish.absolute.spell.train_status = ok
animals.fish.absolute.spell.validationResult = ok
animals.fish.absolute.spell.ValidationRaw.tp = 4
animals.fish.absolute.spell.ValidationRaw.fp = 0
animals.fish.absolute.spell.ValidationRaw.tn = 11
animals.fish.absolute.spell.ValidationRaw.fn = 0
animals.fish.absolute.spell.measure.pred_acc = 1.0
```

If this is not the case, check the logs in the new `sml-tempXXXXXXXXXXXXXXXX` directory (where `XXXXXXXXXXXXXXXX` is some timestamp).

## Reproducing the Strength and Weakness Benchmarks

Inside the SPELL directory (and inside the SPELL virtual environment) run
```
python generate_synthetic.py PATH_TO/sml-bench-core-0.3-SNAPSHOT/learningtasks
```
(replace `PATH_TO` with the appropriate path). This should create the `hard-path`, `hard-conj`, and `hard-deep-conj` benchmarks in the `sml-bench-core-0.3-SNAPSHOT/learningtasks` directory.

Copy the files `benchmarks/bench-k-1-conj.plist`, `benchmarks/bench-k-2-conj.plist`, and `benchmarks/bench-k-path.plist` to the `sml-bench-core-0.3-SNAPSHOT` directory and run the benchmarks with
```
    bin/smlbench bench-k-1-conj.plist 
    bin/smlbench bench-k-2-conj.plist 
    bin/smlbench bench-k-path.plist 
```
This might take some time, as each benchmark has a timeout of 10 minutes.

The results should then be written to `bench-k-1-conj-result.prop`, `bench-k-2-conj-result.prop` and `bench-k-path -result.prop`.

## Reproducing the OWL2Bench Performance Benchmarks

Note that the benchmark generator might not produce the exact benchmarks used for the experiments.

### Obtaining an OWL2Bench Knowledge Base

Easy way: use the file `tests/OWL2EL-1.owl`

Hard way:

Clone the OWL2Bench repository
```
git clone git@github.com:kracr/owl2bench.git
cd owl2bench
git checkout c76187feb4cd008ba1943bd37205a5ade6a3b0d2
```

Generate a knowledge base for the EL profile containing 1 university
```
java --add-opens java.base/java.lang=ALL-UNNAMED -jar OWL2Bench.jar 1 EL
```

Use the resulting `OWL2EL-1.owl` for the next step

### Generating and Running the Benchmarks

Inside the SPELL directory (and inside the SPELL virtual environment) run
```
python generate_owl2bench_perf.py PATH_TO/sml-bench-core-0.3-SNAPSHOT/learningtasks tests/OWL2EL-1.owl
```
(That is one line) Replace `PATH_TO` with the appropriate path. This should create `owl2bench-1` to `owl2bench-6`
in `sml-bench-core-0.3-SNAPSHOT/learningtasks` directory.

Then, copy `benchmarks/bench-owl2bench-perf.plist` to the `sml-bench-core-0.3-SNAPSHOT` directory and run the benchmarks with
```
    bin/smlbench bench-owl2bench-perf.plist 
```

The results should be written to `bench-owl2bench-perf-results.prop`.

## Reproducing the Yago Performance and Generalization Benchmarks

### Obtaining the Yago Fragment

This process takes some time and RAM. You might want to adjust the maximal heap size in `robot/generate_yago_fragment.sh`

Run the `generate_yago_fragment.sh` script inside of the `robot` directory:
```
cd robot
./generate_yago_fragment.sh
```

After this script completes, the yago fragment is available as
`yago-reasoned.owl`. All other files produced files can be deleted.

### Reproducing the Yago Performance Benchmarks

Inside the SPELL directory (and inside the SPELL virtual environment) run
```
python generate_yago_perf.py PATH_TO/sml-bench-core-0.3-SNAPSHOT/learningtasks robot/yago-reasoned.owl
```
(That is one line) Replace `PATH_TO` with the appropriate path. This should create 
`yago-1-succ-reachable-n-k` benchmarks in the in `sml-bench-core-0.3-SNAPSHOT/learningtasks` directory.
The generation process might take some time and RAM. Consider adjusting the maximum heap size for `robot` (which is used during the generation process) at the top of `spell/benchmark_tools.py`.

Then, copy `benchmarks/bench-yago-perf.plist` to the
`sml-bench-core-0.3-SNAPSHOT` directory and run the benchmarks with
```
    bin/smlbench bench-yago-perf.plist 
```
The results should be written to `bench-yago-perf-results.prop`.

### Reproducing the Yago Generalization Benchmarks

Inside the SPELL directory (and inside the SPELL virtual environment) run
```
python generate_yago_gen.py PATH_TO/sml-bench-core-0.3-SNAPSHOT/learningtasks robot/yago-reasoned.owl
```
(That is one line) Replace `PATH_TO` with the appropriate path. This should create 
`yago-gen-test2-n-k` benchmarks in the in `sml-bench-core-0.3-SNAPSHOT/learningtasks` directory.
The generation process might take some time and RAM. Consider adjusting the maximum heap size for `robot` (which is used during the generation process) at the top of `spell/benchmark_tools.py`.

Then, copy `benchmarks/bench-yago-gen.plist` to the
`sml-bench-core-0.3-SNAPSHOT` directory and run the benchmarks with
```
    bin/smlbench bench-yago-gen.plist 
```
The results should be written to `bench-yago-gen-results.prop`.