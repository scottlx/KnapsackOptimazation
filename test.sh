#!/bin/sh
for i in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000
do
  ./Generate $i test2.txt
  ./multiblock
done
