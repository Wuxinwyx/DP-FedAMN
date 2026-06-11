# DP-FedAMN

---

This is the code implementation for paper "DP-FedAMN: Adapting Models to Noise for Improved Utility in Differentially Private Federated Learning".

## Usage

Here is an example to run DP-FedAMN on CIFAR-10 with a simple CNN:

```bash
python main.py --dataset=cifar10 \
    --model=simple-cnn \
    --alg=dpfedamn\
    --lr=0.01 \
    --epochs=10 \
    --comm_round=200 \
    --n_parties=200 \
    --partition=noniid \
    --beta=0.5 \
    --dp_clip=0.8\
    --logdir='./logs/' \
    --datadir='./data/' \
```
