#!/bin/bash

for resource in {15..30}
do
        srun --partition=common --qos=1gpu1h --gres=gpu:1 python3 test_mwu_cp.py $resource $resource 15 3 13 >> tmpcp$resource$field &
done

