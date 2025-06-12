#!/bin/bash


# custom config
DATA="/path/to/datasets"
TRAINER=TAC

CFG=crossdatasets
SHOTS=16

for DATASET in 'imagenet' 'imagenetv2' 'imagenet_sketch' 'imagenet_a' 'imagenet_r' 'caltech101' 'food101' 'dtd' 'ucf101' 'oxford_flowers' 'fgvc_aircraft' 'sun397' 'eurosat' 'oxford_pets' 'stanford_cars'
do
	for SEED in 1 2 3
	do
		DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
		if [ -d "$DIR" ]; then
			echo "Results are available in ${DIR}. Skip this job"
		else
			echo "Run this job and save the output to ${DIR}"

			python train.py \
			--root ${DATA} \
			--seed ${SEED} \
			--trainer ${TRAINER} \
			--dataset-config-file configs/datasets/${DATASET}.yaml \
			--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
			--output-dir ${DIR} \
			--model-dir output/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED} \
			--load-epoch 5 \
			--eval-only
		fi
	done
done

for DATASET in 'imagenet' 'imagenetv2' 'imagenet_sketch' 'imagenet_a' 'imagenet_r' 'caltech101' 'food101' 'dtd' 'ucf101' 'oxford_flowers' 'fgvc_aircraft' 'sun397' 'eurosat' 'oxford_pets' 'stanford_cars'
do 
	python parse_test_res.py output/evaluation/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets_16shots/${DATASET} --test-log
done