#!/bin/bash

# get command line arguments
start=$1
end=$2
batch=$3

cd /home/rcf-proj/ef/spangher/source-exploration/models/ACL2013_Personas/generate_stanford_parses

datadir=../../../data/news-article-flatlist
inputdir=$datadir/raw/$batch
outputdir=$datadir/stanford-parses/$batch
mkdir -p $inputdir
mkdir -p $outputdir

### split up data
echo 'preprocessing using python...'
echo "python3.7 split_a1_file.py --start $start --end $end --batch $batch"
python3.7 split_a1_file.py --start $start --end $end --batch $batch

## aggregate list for CoreNLP
filelist=$datadir/filelist-to-process-batch-$batch.txt
ls -1 -d $datadir/raw/$batch/* > $filelist

echo 'running CoreNLP...'
## run corenlp
java \
	-cp "../stanford-corenlp-full-2018-10-05/*" \
	-Xmx5g \
	edu.stanford.nlp.pipeline.StanfordCoreNLP \
	-annotators tokenize,ssplit,pos,lemma,ner,parse,coref \
	-fileList $filelist
	-outputDirectory $outputdir
