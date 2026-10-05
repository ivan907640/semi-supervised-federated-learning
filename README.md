# Semi-Supervised Federated Learning

Research code for a master's dissertation exploring **semi-supervised federated learning for image classification under non-IID data distributions**.

The project studies a federated setting in which only a small subset of clients have labeled data, while the remaining clients learn from unlabeled data. The implementation combines federated model aggregation with pseudo-labeling, exponential moving average (EMA) teacher models, consistency-based unsupervised learning, and configurable non-IID client partitions.

## Key Ideas

- **Federated learning** with client-side training and global model aggregation
- **Semi-supervised learning** across labeled and unlabeled clients
- **FedAvg-style aggregation** for combining local model updates
- **EMA teacher models** for more stable pseudo-label generation
- **Confidence-thresholded pseudo-labeling**
- **Consistency regularization** using augmented image views
- **Non-IID data partitioning** across participating clients
- Support for experiments with **CIFAR**, **SVHN**, and skin-image classification datasets

## Repository Structure

```text
.
├── train_main.py          # Main federated training pipeline
├── local_supervised.py    # Training logic for labeled clients
├── local_unsupervised.py  # Training logic for unlabeled clients
├── FedAvg.py              # Federated aggregation utilities
├── validation.py          # Evaluation metrics
├── options.py             # Experiment configuration
├── cifar_load.py          # Dataset loading and partitioning
├── datasets.py            # Dataset utilities
├── dataloaders/           # Data-loading helpers
├── networks/              # Model definitions
├── loss/                  # Loss functions
├── partition_strategy/    # Saved client partition configurations
├── warmup/                # Warm-up checkpoints
└── requirements.txt       # Python dependencies
```

## Training Workflow

At a high level, each communication round:

1. Samples a subset of federated clients.
2. Trains labeled clients with supervised objectives.
3. Trains unlabeled clients using EMA-generated pseudo-labels and consistency loss.
4. Aggregates selected client models into the global model.
5. Evaluates and records experiment metrics.

The implementation is designed for experiments where labeled data are scarce and distributed unevenly between clients.

## Installation

Create a Python environment and install the listed dependencies:

```bash
pip install -r requirements.txt
```

The training code uses PyTorch and CUDA-enabled GPUs. Make sure your PyTorch installation matches your CUDA environment.

## Running an Experiment

Experiment settings are defined in `options.py`. Important parameters include:

- `--dataset`: dataset used for training
- `--num_users`: number of federated clients
- `--unsup_num`: number of unlabeled clients
- `--rounds`: communication rounds
- `--local_ep`: local training epochs
- `--partition`: client data-partition strategy
- `--beta`: Dirichlet concentration parameter for non-IID partitioning
- `--confidence-threshold`: pseudo-label confidence threshold
- `--ema_decay`: EMA teacher decay
- `--lambda_u`: unsupervised loss weight

Example:

```bash
python train_main.py --dataset cifar100 --rounds 200 --gpu 0
```

Some experiments rely on pre-generated partition files and checkpoints stored in `partition_strategy/` and `warmup/`.

## Research Context

This repository was developed as part of a master's dissertation investigating how unlabeled client data can be exploited in federated learning when labels are available only on a small fraction of clients.

The code is retained as a research implementation rather than a production framework. Its main purpose is to document the experimental methodology and provide a reproducible reference for the dissertation work.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- scikit-learn
- TensorBoard

## Author

**Ivan**  
Master's dissertation research project.
