#!/bin/sh
module load cuda
for i in 10000 20000 30000 40000 50000 60000 70000 80000 90000 100000
do
  ./Generate $i test2.txt
  ./multiblock
done

