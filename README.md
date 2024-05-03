# IT3001-Federated-Learning-with-Flower 

## Overview: 
- Simulate Federated Learning with Flower & RayIO 


## Installation: 
### Requirements 
- Python 3.11 
- CUDA 12 

### Installation: 
```bash
python -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
```


## Simluate: 
- Centralized training 
```bash 
python train_mobilnetv2_pytorch.py
```

- Decentralized training 
```bash 
python sim_server.py 
```