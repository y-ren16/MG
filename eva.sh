#!/bin/bash

for epoch in {200..1200..100}; do
	for i in {1..5}; do
		python evaluate.py --restore_epoch $epoch -p config/AD/preprocess.yaml -m config/AD/model.yaml -t config/AD/train.yaml --time_dir 2023-03-25-23_50 | pv -L 5k >> /dev/null
	done
	echo "Epoch $epoch complete"
done

