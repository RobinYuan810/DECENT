#!/bin/bash
# eg：bash run.bash DECENT GCN


# datasets
DATASETS=("Cora" "CiteSeer" "PubMed" "CoraFull" "Computers" "Photo" "CS" )

#DATASETS=("PubMed")
MODEL_NAME="$1"
BACKBONE="$2"

BASE_NUM=10
CALIB_NUM=5
GPU_DEVICE=4

declare -A alpha_values
alpha_values=(
    ["Cora"]="0.03"
    ["CiteSeer"]="0.01"
    ["PubMed"]="0.09"
    ["CoraFull"]="0.09"
    ["Computers"]="0.09"
    ["Photo"]="0.09"
    ["CS"]="0.01"
)

declare -A gamma_values
gamma_values=(
    ["Cora"]="0.07"
    ["CiteSeer"]="0.09"
    ["PubMed"]="0.01"
    ["CoraFull"]="0.01"
    ["Computers"]="0.05"
    ["Photo"]="0.01"
    ["CS"]="0.09"
)

for DATASET in "${DATASETS[@]}"; do
    CSV_FILE="results/${MODEL_NAME}/${BACKBONE}/${DATASET}/ret.csv"
    > "$CSV_FILE"
    echo "${CSV_FILE} has been cleared."

    ALPHA="${alpha_values[$DATASET]}"
    GAMMA="${gamma_values[$DATASET]}"

    python main.py --modelname="$MODEL_NAME" --backbone="$BACKBONE" --dataset="$DATASET" --base_num="$BASE_NUM" --calib_num="$CALIB_NUM" --gpu_device="$GPU_DEVICE" --alpha="$ALPHA" --gamma="$GAMMA"
done

