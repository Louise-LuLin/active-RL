#!/bin/bash
read -p "Enter source, budget, greedy, replaceX, word2vec, cuda: " s b g x w e
echo
reward=("valid2T" "valid2V" "test2T" "kmers")
for r in ${reward[@]}
do
    python run.py -s $s -b $b -r $r -g $g -x $x -w $w -e $e;
done
