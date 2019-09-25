#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH -t 24:00:00

cd /home/rcf-proj/ef/spangher/source-exploration/models/ACL2013_Personas/java

#!/bin/bash
#####################################################
numtopics=5
maxvocab=1000
numpersonas=50
runreg=true

#name=$1
#input=$2
name=newspapers__numtopics-$numtopics__numpersonas-$numpersonas__runreg-$runreg
input=../../../data/news-article-flatlist/preprocessed/all-data.data
heapsize=3000m

echo "output dir: $name"

mkdir -p "$name.out"

cat > "$name.properties"  <<-EOF
# number of topics
K=$numtopics
# number of personas
A=$numpersonas
# max vocabulary size
V=$maxvocab

# initial dirichlet document-persona smoother (this gets optimized)
alpha=10

# initial dirichlet topic-word smoother (this gets optimized)
gamma=1

# L2 regularization parameter (for Persona Regression model)
L2=.01

# max number of iterations
maxIterations=50000

# true = run Persona Regression model; false = run Dirichlet Persona Model.
runPersonaRegressionModel=$runreg

# input
data=$input
# movieMetadata=input/all.movies.metadata
# characterMetadata=input/all.character.metadata

# output
characterPosteriorFile="$name.out"/25.100.lda.log.txt
characterConditionalPosteriorFile="$name.out"/25.100.lda.cond.log.txt
outPhiWeights="$name.out"/out.phi.weights
weights="$name.out"/lr.weights.txt
featureMeans="$name.out"/featureMeans.txt
featureFile="$name.out"/featureFile.txt

personaFile="$name.out"/personaFile
finalLAgentsFile="$name.out"/finalLAgentsFile
finalLPatientsFile="$name.out"/finalLPatientsFile
finalLModFile="$name.out"/finalLModFile
EOF

# add all the jars anywhere in the lib/ directory to our classpath
here=$(dirname runjava)
CLASSES=$here/out/production/java
CLASSES=$CLASSES:$(echo $here/lib/*.jar | tr ' ' :)
CLASSES=$CLASSES:$here/narrative.jar

echo "COMMAND: java.exe -XX:ParallelGCThreads=2 -Xmx$heapsize -Xms$heapsize -ea -classpath $CLASSES $*" personas.ark.cs.cmu.edu/PersonaModel "$name.properties"
java -XX:ParallelGCThreads=2 -Xmx$heapsize -Xms$heapsize -ea -classpath $CLASSES personas.ark.cs.cmu.edu/PersonaModel "$name.properties"
