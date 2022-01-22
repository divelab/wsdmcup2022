# WSDM2022
DIVE@TAMU Solution for WSDM2022



## Requirement

numpy == 1.21.2
pandas == 1.3.5
torch == 1.10.1
torchaudio == 0.10.1
torchvision == 0.11.2
tqdm == 4.62.3



## Experiment

Put all the csv data in a directory. Rename the `data_path` in `reindex_A.py` as the target data directory. Then run `reindex_A.py` first. This is a file that re-index the node of dataset A.

We provide two shell files: `runA.sh` and `runB.sh`, and to run experiment on dataset A and B, we need to additionally specify data directory `--dataset_path xxx` in each shell file first.



## Testing Results in Intermediate testing dataset 

|      | A    | B    |
| ---- | ---- | ---- |
| AUC  | 0.58 | 0.86 |

