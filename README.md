# DP-FedAMN

This is the code implementation for paper "DP-FedAMN: Adapting Models to Noise for Improved Utility in Differentially Private Federated Learning".

**Abstract:** Federated learning (FL) has become a promising paradigm for privacy-preserving data analytics, as it allows multiple clients to collaboratively train a shared model over decentralized data without exposing their local data. To further strengthen privacy and safeguard clients' private information against inference attacks, client-level differential privacy (DP) is widely adopted in FL to provide rigorous privacy guarantees. However, existing methods for ensuring client-level DP inevitably incur noise-induced bias, which causes local updates to deviate from the global optimization direction and results in significantly performance degradation. Furthermore, in heterogeneous data scenarios, the injected noise may exacerbate local update drift in FL. To address these issues, we propose DP-FedAMN, a novel differentially private federated learning (DPFL) scheme designed to adapt local models to noise. Our key idea is to incorporate DP noise as an integral part of the optimized local models, thereby fundamentally mitigating noise-induced bias. Specifically, we design two new techniques, DP noise adaptation and local update norm constraint, where the former guides noisy local models converge to local optima and alleviates model drift, while the latter prevents information loss due to clipping. We also provide formal client-level DP guarantees and an improved convergence bound, and theoretically analyze how our method mitigates the noise-induced bias. Extensive experiments demonstrate that DP-FedAMN outperforms state-of-the-art DPFL schemes, 
improving model accuracy by up to 2.38% while preserving comparable efficiency and privacy protection.

## Usage

The following command provides an example of running DP-FedAMN with a simple CNN on CIFAR-10.

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
