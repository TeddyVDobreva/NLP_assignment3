The repository has the following structure:

```
.
├── data
│   ├── test.csv
│   └── train.csv
├── main.py
├── notebook
│   └── NLP3.ipynb
├── plots
│   ├── confusion_matrix_headlines_test.png
│   ├── confusion_matrix_mask_dataset.png
│   ├── confusion_matrix_test_dataset.png
│   └── confusion_matrix_validation_dataset.png
├── pyproject.toml
├── README.md
├── src
│   ├── __pycache__
│   │   ├── data_handler.cpython-312.pyc
│   │   └── evaluation.cpython-312.pyc
│   ├── data_handler.py
│   └── evaluation.py
└── uv.lock
```

You can run the code by following the instructions enclosed here.

First, clone the repository by running:

- for SSH

'''
git clone git@github.com:TeddyVDobreva/NLP_assignment2.git
'''

- for HTTPS

'''
git clone https://github.com/TeddyVDobreva/NLP_assignment2.git
'''

After cloning the repository, you want to activate the uv environement.

'''
uv sync
uv source .venv/bin/activate
'''

After this, you are ready run the code in the terminal:

'''
python main.py
'''