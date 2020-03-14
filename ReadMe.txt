# Setup

Install requirements:
```
pip isntall -r requirements.txt
```
Install requirements for e2e metrics:
```
pip install -r e2e_metrics/requirements
```

# Execute

Trained personality classifier (change on preprocessing), re-train classifier again before train a generation model
```
python model.py -domain personage -network classifier -mode [train | adjust | test]
```

Example command:
```
python model.py -domain personage -network controlled_VQVAE  -mode train
```

The file `config.py` contains the configuration parameters.


# TODO

Lei: add comments on hyperparameters
