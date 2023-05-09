
wget "https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-schema.nt.gz"

wget "https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-class.nt.gz"

wget "https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-facts.nt.gz"

wget "https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-simple-types.nt.gz"

wget "https://yago-knowledge.org/data/yago4/en/2020-02-24/yago-wd-full-types.nt.gz"

export ROBOT_JAVA_ARGS="-Xmx40G -XX:+UseShenandoahGC"
export LC_ALL=en_US.UTF-8


./robot merge -vvv --input yago-wd-schema.nt.gz --input yago-wd-facts.nt.gz --output yago-facts.owl

./robot filter -vvv -input yago-facts.owl --term-file persons.txt --trim false --select "self types" --axioms "ObjectPropertyAssertion" --output yago-facts-filtered.owl

cat yago-facts-filtered.owl | grep "<NamedIndividual" | grep -o "\".*\"" | tr -d "\"" > persons2.txt

sed -i "" "s/&apos;/'/g" persons2.txt
sed -i "" "s/\&amp;/\&/g" persons2.txt

./robot filter -vvv -input yago-facts.owl --term-file persons2.txt --trim false --select "self types" --axioms "ObjectPropertyAssertion" --output yago-facts-filtered.owl

cat yago-facts-filtered.owl | grep "<NamedIndividual" | grep -o "\".*\"" | tr -d "\"" > persons3.txt

sed -i "" "s/&apos;/'/g" persons3.txt
sed -i "" "s/\&amp;/\&/g" persons3.txt

./robot -vvv merge --input yago-wd-schema.nt.gz --input yago-wd-simple-types.nt.gz filter --term-file persons3.txt --trim false --output yago-st-filtered.owl

./robot -vvv merge  --input yago-wd-schema.nt.gz --input yago-wd-full-types.nt.gz filter --term-file persons3.txt --trim false --output yago-ft-filtered.owl

./robot -vvv merge --input yago-wd-schema.nt.gz --input yago-wd-class.nt.gz --input yago-facts-filtered.owl --input yago-st-filtered.owl --input yago-ft-filtered.owl --output yago-filtered.owl

./robot -vvv remove --input yago-filtered.owl --term-file unsat-classes.txt --output yago-sat.owl

# Get rid of disjointness axioms since otherwise yago is incosistent 
./robot -vvv remove --input yago-filtered.owl --axioms disjoint --output yago-sat.owl

./robot -vvv reason --input yago-sat.owl --include-indirect true --axiom-generators "ClassAssertion PropertyAssertion" --output yago-reasoned.owl

# Rename some individuals as they make problems during the following steps otherwise
sed -i "" "s%geo:%http://geo.com#%g" yago-reasoned.owl

