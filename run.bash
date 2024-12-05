#!/bin/bash
# 启动方式eg：bash run.bash DECENT GCN
# bash run.bash DECENT TAGCN
# bash run.bash DECENT SGC
# bash run.bash DECENT GraphSAGE
# 单独的eg:python main.py --modelname="DECENT" --backbone="GCN" --dataset='Cora' --base_num=20 --calib_num=5 --gpu_device=4

# datasets 列表中定义要使用的所有数据集
#DATASETS=("Cora" "CiteSeer" "PubMed" "CoraFull" "Computers" "Photo" "CS" "Physics"
# "Cornell" "Texas" "Wisconsin" "Actor" "Chameleon" "Squirrel")
#DATASETS=("Cora" "CiteSeer" "PubMed" "CoraFull" "Computers" "Photo" "CS" )
DATASETS=("Cora" "CiteSeer" "PubMed" "Photo" )
#DATASETS=("PubMed")
MODEL_NAME="$1"
BACKBONE="$2"

BASE_NUM=10
CALIB_NUM=5
GPU_DEVICE=4

# 定义数据集和对应的alpha值的关联数组
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

# 定义数据集和对应的gamma值的关联数组
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

# 循环遍历每个数据集
for DATASET in "${DATASETS[@]}"; do
    # 定义 CSV 文件路径
    CSV_FILE="results/${MODEL_NAME}/${BACKBONE}/${DATASET}/ret.csv"

    # 清空或创建 CSV 文件
    > "$CSV_FILE"
    echo "${CSV_FILE} has been cleared."
#--ew_method="Transformer"
    # 获取当前数据集的alpha值和gamma值
    ALPHA="${alpha_values[$DATASET]}"
    GAMMA="${gamma_values[$DATASET]}"
    # 运行 Python 脚本
    python main.py --modelname="$MODEL_NAME" --backbone="$BACKBONE" --dataset="$DATASET" --base_num="$BASE_NUM" --calib_num="$CALIB_NUM" --gpu_device="$GPU_DEVICE" --alpha="$ALPHA" --gamma="$GAMMA"
done

