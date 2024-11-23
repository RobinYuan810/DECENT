# DECENT: Data-Centric Edge and Node Tuning for GNN Calibration

You can get the results in the paper by running main.py and modifying the parameters in the code

## 我的工作：

DECENT:模型的完全体，包括两个核心模块NTM和ETM
how to start

# 一、服务器激活环境，切换目录

conda activate /home/training_center/zlf_envs/DL
cd /home/training_center/code/ybc/DECENT/

# 二、超参数分析

跑"Cora" "CiteSeer" "PubMed" "CoraFull" "Computers" "Photo" "CS" "Physics"
--if_draw='no'，不用画任何结果图(因为我们只希望最好超参数组合)
bash run_hyper.bash DECENT_hyper GCN
bash run_hyper2.bash DECENT_hyper GCN
bash run_hyper3.bash DECENT_hyper GCN
bash run_hyper4.bash DECENT_hyper GCN
bash run_hyper5.bash DECENT_hyper GCN
bash run_hyper6.bash DECENT_hyper GCN
bash run_hyper7.bash DECENT_hyper GCN
bash run_hyper8.bash DECENT_hyper GCN

## 超参数分析的结果

| Dataset   | Train_Ratio | Alpha | Gamma | Notes        |
|-----------|-------------|-------|-------|--------------|
| Cora      | 0.6         | 0.03  | 0.07  | 0.206        |
| CiteSeer  | 0.6         | 0.01  | 0.09  | 0.26         |
| PubMed    | 0.6         | 0.09  | 0.01  | 0.46         |
| CoraFull  | 0.6         | 0.09  | 0.03  | 0.51         |
| Computers | 0.6         | 0.07  | 0.07  | 0.26         |
| Photo     | 0.6         | 0.09  | 0.01  | 0.51         |
| CS        | 0.6         | 0.01  | 0.09  | 0.41         |

## 绘制超参数分析的图

# 三、正常的DECENT

bash run.bash DECENT GCN
bash run2.bash DECENT GCN
bash run3.bash DECENT GCN

# 四、消融实验

bash run_ablation.bash DECENT_ablation GCN



#   D E C E N T  
 