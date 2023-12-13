# Mixture of Experts & Synthetic Data
Simple implementation of the mixture-of-experts architecture combined with a simple framework for generating synthetic data.
This repository is a simple combinations of Andrej Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT) implementation and Philip Wang's implementation of Mixture-of-Experts (https://github.com/lucidrains/mixture-of-experts) using PyTorch.


## Repository Structure
moe
 ├── config
 ├── data
 │    └── shakespeare_char
 ├── model
 │   ├── Gates
 │   │    ├── distributed.py
 │   │    └── stmoe.py
 │   ├── MLPExpert.py
 │   └── model.py
 ├── out
 ├── train.py
 └──requirements.txt
synth
 ├── features
 ├── samples
 └──synthesize_data.py

 ## Running a model
 All training is done using the train.py script. 
Install requirements:

```
pip install -r requirements.txt
```

Run the training script in your terminal:

```
python train.py config/train_shakespeare_char.py
```

This will run a pre-specified model with the hyperparameters specified in train_shakespeare_char.py. It is also possible to overwrite hyperparameters from the command-line. See the configurator.py for more details.