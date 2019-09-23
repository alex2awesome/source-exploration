#!/bin/bash

datadir=../../../data/news-article-flatlist
filelist=$datadir/filelist-to-process.txt
ls -1 $datadir/raw > $filelist

java.exe \
	-cp "../stanford-corenlp-full-2018-10-05/*" \
	-Xmx2g \
	edu.stanford.nlp.pipeline.StanfordCoreNLP \
	-annotators tokenize,ssplit,pos,lemma,ner,parse,coref \
	-fileList $filelist
