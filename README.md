# WSDM2022
DIVE@TAMU Solution for WSDM2022



## Requirement

`pip install -r requirements.txt`



## Experiment

Put all the csv data in a directory. Rename the `data_path` in `reindex_A.py` as the target data directory. Then run `reindex_A.py` first. This is a file that re-index the node of dataset A.

We provide two shell files: `runA.sh` and `runB.sh`, and to run experiment on dataset A and B, we need to additionally specify data directory `--dataset_path xxx` in each shell file first.



## Testing Results in Intermediate Testing Dataset 

|      | A    | B    |
| ---- | ---- | ---- |
| AUC  | 0.58 | 0.86 |

