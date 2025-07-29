# ManyProp
A simple machine learning project made for an internship at NIST. Predicts a single chemical property of a mixture of two or more chemicals. Models and data are not included.  

## Installation
___
- `git clone https://github.com/Arrowshark1/ManyProp.git` 
- `cd ManyProp
- `python -m venv ./manypropenv`
- `cd manypropenv` 
- `.\manypropenv\Scripts\activate`
- `pip install -r .\requirements.txt`
    - use `requirements-no-cuda.txt` if your computer lacks a Nvidia GPU
## Usage
___
- to run use `python .\ManyProp.py --smiles MOL1 MOL2 ... --mol_fracs FRAC1 FRAC2 ... --checkpoints_dir PATH_TO_CHECKPOINTS --num_mols NUMBER_OF_MOLECULES`
    - requires models to be present in user provided checkpoints path
    - number of fractions provided must either equal number of molecules or one minus the number of molecules 
- to train on a dataset use `python .\ManyProp.py --train True --data_path PATH_TO_DATA --mol_features_path PATH_TO_FEATURES --num_mols NUMBER_OF_MOLECULES --smiles_columns COL1 COL2 ... --targets_column TGT_COL --mol_features_columns FEAT1 FEAT2 ... --mol_frac_columns FRAC1 FRAC2 ...`
    - dataset must come in the form of a .csv file
    - can only contain specified columns 