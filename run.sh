#!/bin/bash
read -p "Enter source, budget, model, greedy, replaceX, word2vec, cuda: " s b m g x w e
echo
reward=("valid2V")
for r in ${reward[@]}
do
    python run_trellis.py -s $s -b $b -m $m -r $r -g $g -x $x -w $w -e $e;
done
