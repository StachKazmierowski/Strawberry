#!/bin/bash
steps_number=10000
phi=0.125

cd src

for troopsANo in 8 9 10
do
    for troopsBNo in 8 9 10
    do
        for battlefieldsNo in 5 6 7
        do
        	python3 main.py $troopsANo $troopsBNo $battlefieldsNo $steps_number $phi
        done
    done
done

cd ..
