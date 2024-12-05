# DECENT: Data-Centric Edge and Node Tuning for GNN Calibration

To protect our research, we have not disclosed the complete codebase. Currently, only a portion of the code that demonstrates our methodology is available. Once the paper is accepted and indexed, all the code will be available.

You can get the results in the paper by running main.py and modifying the parameters in the code

## My Work

DECENT: The Complete Model, Including Two Core Modules NTM and ETM

# 1. Server Activation Environment, Change Directory


# 2. Hyperparameter Analysis

Run "Cora", "CiteSeer", "PubMed", "CoraFull", "Computers", "Photo", "CS", "Physics".
`--if_draw='no'`, do not draw any result graphs (since we only want the best hyperparameter combination).
bash run_hyper.bash DECENT_hyper GCN

## Hyperparameter Analysis Results

| Dataset   | Train_Ratio | Alpha | Gamma | Notes |
|-----------|-------------|-------|-------|-------|
| Cora      | 0.6         | 0.03  | 0.07  | 0.206 |
| CiteSeer  | 0.6         | 0.01  | 0.09  | 0.26  |
| PubMed    | 0.6         | 0.09  | 0.01  | 0.46  |
| CoraFull  | 0.6         | 0.09  | 0.03  | 0.51  |
| Computers | 0.6         | 0.07  | 0.07  | 0.26  |
| Photo     | 0.6         | 0.09  | 0.01  | 0.51  |
| CS        | 0.6         | 0.01  | 0.09  | 0.41  |

## Plotting Hyperparameter Analysis Graphs

# 3. Normal DECENT

bash run.bash DECENT GCN

# 4. Ablation Study

bash run_ablation.bash DECENT_ablation GCN