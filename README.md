# Active Learning for Multi-Class Drug-Drug Interactions Prediction

## Introduction
We combine active learning with the multi-class DDIs prediction for the first time to reducing DDIs annotation costs while maintaining the models' performance. In our work, we propose a novel active sampling strategy named Margin-based Dynamic Cluster tailored for the DDIs datasets.
![截屏2023-09-26 14 21 43](https://github.com/pantherang/ALDDI/assets/49769931/6bbac916-8c7f-4bbc-a727-0cefbc1cabc1)

## Requirements
To run the code, you need the following dependencies:
- numpy == 1.21.6
- Python == 3.7.13
- PyTorch == 1.13.1+cu117
- PyTorch Geometry == 2.3.0
- rdkit == 2020.09.1
- pandas == 1.3.5
- scikit-learn == 1.0.2

## Step-by-step running
- Firstly, you should use ```python cross-validation.py``` to split the original DrugBank dataset.
- Then, to specify various settings, such as sampling strategies and drug pair encoders, you will need to modify the ```train_config.py``` file.
- After modifying the parameters, you can use ```python train.py``` to run the combination of drug encoders and sampling strategies.

## Supplementary explanation
If you are interested in prior DDIs research, you can access their GitHub repositories via the following links:
- SA-DDI: https://github.com/guaguabujianle/SA-DDI
- SSI-DDI: https://github.com/kanz76/SSI-DDI
- GMPNN-CS: https://github.com/kanz76/GMPNN-CS
- DSN-DDI: https://github.com/microsoft/Drug-Interaction-Research/tree/DSN-DDI-for-DDI-Prediction
- DGNN-DDI: https://github.com/mamei1016/DGNN-DDI
