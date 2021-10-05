#!/bin/bash

for resource in {15..25}
do
	python3 test_mwu.py $resource 10 13 0 >> tmp$resource$field &
done

for resource in {26..30}
do
	python3 test_mwu.py $resource 10 13 15 >> tmp$resource15 &
done
