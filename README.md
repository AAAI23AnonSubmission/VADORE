# Vadore


Repository for the experiments on the RecSys 2017 Challenge dataset in the context of the AAAI 2023 submission "A Job Recommender System in Covid Times".

## Requirements

The version of the packages used for the computational experiments are described in the requirements.txt file.

To setup a virtual environment and download the required packages, you may for instance use the `virtualenv` module from the python standard 
library and follow the next instructions:


Create a virtualenv named `venv`

```
python3 -m venv venv
```


Activate it

```
source bin/activate
```


Install packages from the requirements file

```
pip3 install -r requirements.txt
```

## Replication of the experiments

Download the data files from the DropoutNet (Volkovs et al., 2017) Paper github (i.e. https://github.com/layer6ai-labs/DropoutNet); extract them to form a recsys2017.pub/ folder. 

Train the Vad.0 module (saving the Vad.0 standard scalers in directory vad0):
```
python train_vad0.py recsys2017.pub/ vad0
```

Save the Vad.0 top-1k ranks for each of the train uids with matches (in a file named top.csv):
```
python select_top.py recsys2017.pub/ vad0 top.csv
```

Train the Vad.1 module (saving the models in file vad1):
```
python train_vad1.py recsys2017.pub/ top.csv vad1
```

Compute ranks (those less than 100, those above 100 are turned into Inf to avoid exhaustive sorting of recommendations) for the Vad.1 module, and save them:

* In the user cold start case (saving recommendations to user_cold_ranks.csv):
```
python eval_vad1.py recsys2017.pub/ vad0 vad1 dates.csv  user_cold user_cold_ranks.csv
```
* In the warm start case  (saving recommendations to warm_ranks.csv)::
```
python eval_vad1.py recsys2017.pub/ vad0 vad1 dates.csv warm warm_ranks.csv
```