# Final Project - Big Data

## Set Up
```
conda create --name your_env_name python=3.10
conda activate your_env_name
pip install pandas
```

## Submission Format
Please provide two files:

- public_submission.csv for evaluation on the public dataset

- private_submission.csv for evaluation on the private dataset

Each file should follow the format below:
```
id,label
0,1
1,2
2,3
...
```

#### id should follow the original sample order. label is your predicted value.


## Validation
Use eval.py to check your performance on the public dataset:
``
python eval.py
``