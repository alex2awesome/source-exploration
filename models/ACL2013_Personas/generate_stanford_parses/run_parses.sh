#!/bin/bash

datadir=../../../data/news-article-flatlist
filelist=$datadir/filelist-to-process.txt
ls -1 $datadir/raw > $filelist

java.exe \
	-cp "../java/lib" \
	edu.stanford.nlp.pipeline.StanfordCoreNLP \
	-annotators tokenize,ssplit,pos,lemma,ner,parse,coref \
	-fileList $filelist
