#!/bin/bash


# custom config
DATA="/path/to/datasets"
TRAINER=TAC

DATASET=$1

CFG=basetonovel
SHOTS=16
LOADEP=20
SUB=new

for SEED in 1 2 3
do
	COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
	MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
	DIR=output/base2new/test_${SUB}/${COMMON_DIR}
	if [ -d "$DIR" ]; then
		echo "Evaluating model"
		echo "Results are available in ${DIR}. Resuming..."

		python train.py \
		--root ${DATA} \
		--seed ${SEED} \
		--trainer ${TRAINER} \
		--dataset-config-file configs/datasets/${DATASET}.yaml \
		--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
		--output-dir ${DIR} \
		--model-dir ${MODEL_DIR} \
		--load-epoch ${LOADEP} \
		--eval-only \
		DATASET.NUM_SHOTS ${SHOTS} \
		DATASET.SUBSAMPLE_CLASSES ${SUB}

	else
		echo "Evaluating model"
		echo "Runing the first phase job and save the output to ${DIR}"

		python train.py \
		--root ${DATA} \
		--seed ${SEED} \
		--trainer ${TRAINER} \
		--dataset-config-file configs/datasets/${DATASET}.yaml \
		--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
		--output-dir ${DIR} \
		--model-dir ${MODEL_DIR} \
		--load-epoch ${LOADEP} \
		--eval-only \
		DATASET.NUM_SHOTS ${SHOTS} \
		DATASET.SUBSAMPLE_CLASSES ${SUB}
	fi
done

python parse_test_res.py output/base2new/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG} --test-log