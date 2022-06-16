'''
Created on Dec 19, 2012

Transform coreproc'd files to slim format, adding canonical freebase
character ids where they can be aligned.

@author: david
'''

import sys,re,json,operator,random

def main(lines):
    agents=[]
    patients=[]
    modifiers = []

    fbs={}
    fbactors={}

    docID=0

    counts={}

    enames={}
    fullnames={}
    sstags={}
    for line in lines: 
        if re.search(r'^E\d+\t', line):
            cols=line.rstrip().split("\t")
            if len(cols) > 1:
                id=cols[0].lower()
                info = json.loads(cols[1])
                info['fb'] = info.get('fb','')
                info['fba'] = info.get('fba','')

                fb=info['fb']
                fbactor=info['fba']

                if fb == "":
                    fb=id
                    fbactor=id

                fbs[id]=fb
                fbactors[id]=fbactor

                mdict=info['lemma_c']
                sorted_mdict = sorted(iter(mdict.items()), key=operator.itemgetter(1), reverse=True)
                count=0

                for name,count in sorted_mdict:
                    enames[fb]=name
                    break

                mdict=info['fulltext_c']
                sorted_mdict = sorted(iter(mdict.items()), key=operator.itemgetter(1), reverse=True)
                count=0

                for name,count in sorted_mdict:
                    fullnames[fb]=name
                    break

                mdict=info['sstag_c']
                sorted_mdict = sorted(iter(mdict.items()), key=operator.itemgetter(1), reverse=True)
                count=0

                sstags[fb]=""

                for name,count in sorted_mdict:
                    sstags[fb]=name
                    break

    for line in lines:
        if line.startswith("=== DOC"):
            cols=line.rstrip().split(" ")
            docID=cols[2]
        if re.search(r'^(T|M)[^ ]*\t', line):
            cols=line.rstrip().split("\t")
            tupleID=cols[0]
            main=cols[1]
            parts=main.split(" ")

            if len(parts) > 3:
                verb=parts[0]
                supersense=parts[1]
                rel=parts[2]
                entityID=parts[3]
                entity=parts[4]
                
                eID=entityID.lower()
                fbEntity=fbs[eID]

                sstag=sstags[fbEntity]
                if sstag != "noun.person":
                    continue

                key="%s:%s:%s:%s:%s" % (fbEntity, tupleID, supersense, verb, rel)
                
                count=0
                if fbEntity not in counts:
                    counts[fbEntity]=0
                counts[fbEntity]+=1

                if rel.startswith("A:"):
                    agents.append(key)
                elif rel.startswith("P:"):
                    patients.append(key)
                elif rel.startswith("M:"):
                    modifiers.append(key)
                else: assert False

    print("%s\t" % docID, end=' ')
    for key in agents:
        cols=key.split(":")
        entityID=cols[0]
        if counts[entityID] > 0:
            print("%s" % key.lower(), end=' ')

    print("\t", end=' ')
    for key in patients:
        cols=key.split(":")
        entityID=cols[0]
        if counts[entityID] > 0:
            print("%s" % key.lower(), end=' ')

    print("\t", end=' ')
    for key in modifiers:
        cols=key.split(":")
        entityID=cols[0]
        if counts[entityID] > 0:
            print("%s" % key.lower(), end=' ')

    print("\t", end=' ')
    print(json.dumps(enames), end=' ')
    print("\t", end=' ')
    print(json.dumps(fullnames))

def yield_docs():
    cur = []
    for line in sys.stdin:
        if line.startswith('=== DOC'):
            if cur:
                yield cur
            cur = []
        if line.strip():
            cur.append(line)
    yield cur

if __name__ == "__main__":
    for doclines in yield_docs():
        main(doclines)
