#!/bin/bash

# custom config
DATA="/path/to/datasets"
TRAINER=TAC

CFG=basetonovel
SHOTS=16
SUB=new

for DATASET in 'imagenet' 'caltech101' 'oxford_pets' 'stanford_cars' 'oxford_flowers' 'food101' 'fgvc_aircraft' 'sun397' 'dtd' 'eurosat' 'ucf101'
do
	for SEED in 1 2 3
	do
		DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
		if [ -d "$DIR" ]; then
			echo "Results are available in ${DIR}. Resuming..."
			python train.py \
			--root ${DATA} \
			--seed ${SEED} \
			--trainer ${TRAINER} \
			--dataset-config-file configs/datasets/${DATASET}.yaml \
			--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
			--output-dir ${DIR} \
			DATASET.NUM_SHOTS ${SHOTS} \
			DATASET.SUBSAMPLE_CLASSES base
		else
			echo "Run this job and save the output to ${DIR}"
			python train.py \
			--root ${DATA} \
			--seed ${SEED} \
			--trainer ${TRAINER} \
			--dataset-config-file configs/datasets/${DATASET}.yaml \
			--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
			--output-dir ${DIR} \
			DATASET.NUM_SHOTS ${SHOTS} \
			DATASET.SUBSAMPLE_CLASSES base
		fi
	done
	
	python parse_test_res.py output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}

	bash scripts/tac/base2new_test.sh ${DATASET}
done

for DATASET in 'imagenet' 'caltech101' 'oxford_pets' 'stanford_cars' 'oxford_flowers' 'food101' 'fgvc_aircraft' 'sun397' 'dtd' 'eurosat' 'ucf101'
do
	python parse_test_res.py output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
	python parse_test_res.py output/base2new/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG} --test-log
done
