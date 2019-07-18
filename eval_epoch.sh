#!/bin/bash
echo "we will evaluate epoch from 1 to 10"
for epoch in 1 2 3 4 5 6 7 8 9 10
do
	echo "Train the-epoch:"  $epoch
	python PCNN_SATT.py -data data/person_processed.pkl -name pcnn  -epoch=$epoch -restore_epoch=0 -only_eval


	echo "---------------------------------------------\n" >> ./eval_by_epoch.txt
done
echo "ALl epoch is saved to eval_by_epoch file"