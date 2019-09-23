#!/bin/bash

# get command line arguments
while getopts ":s:e:b" opt; do
  case $opt in
    s) start="$OPTARG"
    ;;
    e) end="$OPTARG"
    ;;
    b) batch="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

datadir=../../../data/news-article-flatlist
outputdir=$datadir/stanford-parses/$batch
mkdir -p $outputdir

### split up data
echo 'preprocessing using python...'
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
