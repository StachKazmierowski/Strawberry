#!/bin/bash


for resource in {15..30}
do
	srun python3 test_mwu_cp.py $resource $resource 15 10 13 >> cptmp$resource &
done
