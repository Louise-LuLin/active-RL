#!/bin/bash
read -p "Enter source, pretrain size, budget, greedy, loop, loop-step, loop-add, replaceX, word2vec, cuda: " s a b g l p u x w e
echo
reward=("valid2T" "valid2V" "test2T" "valid3T" "valid3V" "test3T" "kmers")
for r in ${reward[@]}
do
    python dqn_seq.py -s $s -a $a -b $b -r $r -g $g -l $l -p $p -u $u -x $x -w $w -e $e;
done
