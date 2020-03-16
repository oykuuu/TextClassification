# Category Predictor

The goal of the CategoryPredictor is to accurately predict the multiple category labels of a company given a short description of the company. To this end, the predictor follows a data-driven Artificial Intelligence based solution to analyze the company descriptions and make category predictions.

There are 3 main components in the model. These are pretrained FastText word embeddings, LSTM-based recurrent neural network and label similarity. The recurrent network and the label similarity can be thought of as two different models tasked with the same category prediction job. Both models use the word embeddings when extracting information to accomplish this task. In the end, the information extracted from the recurrent network and label similarity are represented in a tensor which are concatenated and passed through a single feedforward linear layer. 

Below is the model architecture of the CategoryPredictor.

![model_arch](https://user-images.githubusercontent.com/33582865/76808913-81a96c00-67bf-11ea-84cb-f1fb1dabfdb8.png)


## Environment Setup

The conda environment can be recreated using the following command.

`conda create -f environment.yaml`

To use the conda environment in a Jupyter notebook, run:

`python -m ipykernel install --user --name text_env --display-name "Python (text_env)"`
