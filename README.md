# Multitask Recommender System

This repository implements a Multi-Task Neural Network for Recommender Systems using PyTorch. The model is designed to handle both implicit feedback (via Matrix Factorization) and explicit feedback (via Regression) simultaneously. It utilizes the MovieLens dataset for training and evaluation.

## Features

*   **Multi-Task Learning**: Combines Matrix Factorization (BPR loss) and Regression (MSE loss) in a single model.
*   **Flexible Architecture**: Supports shared or separate embeddings for the factorization and regression tasks.
*   **MovieLens Integration**: Automatically downloads and processes MovieLens datasets (default: 100K).
*   **Evaluation Metrics**: Tracks Mean Reciprocal Rank (MRR) and Mean Squared Error (MSE).
*   **TensorBoard Support**: Logs training metrics for visualization.

## Setup

Ensure you have Python installed. You can install the required dependencies using pip:

```bash
python -m pip install -r requirements.txt
```

or by creating a Virtual Envirmoment and installing the dependencies by running the shell script (Recommended).

```bash
bash setup.sh
```

## Usage

To train the model, run the `src/main.py` script. You can configure the training process using command-line arguments.

### Basic Example

```bash
python src/main.py --epochs 10
```

### Custom Configuration

```bash
python src/main.py \
    --logdir logs/experiment1 \
    --test_fraction 0.2 \
    --shared_embeddings \
    --factorization_weight 0.5 \
    --regression_weight 0.5 \
    --epochs 20
```

## Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--test_fraction` | float | `0.05` | Fraction of the dataset to use for testing. |
| `--epochs` | int | `1` | Number of training epochs. |
| `--factorization_weight` | float | `0.995` | Weight for the factorization task loss. |
| `--regression_weight` | float | `0.005` | Weight for the regression task loss. |
| `--shared_embeddings` | flag | `True` | Use shared embeddings for both tasks. |
| `--no_shared_embeddings` | flag | `False` | Use separate embeddings for factorization and regression. |
| `--logdir` | str | `run/...` | Directory for TensorBoard logs. |
| `--gpu` | bool | `False` | Enable GPU training. |

## Project Structure

*   `src/main.py`: Entry point for training the model.
*   `src/multitask.py`: Implements the `MultitaskModel` class handling the training loop.
*   `src/models.py`: Defines the `MultiTaskNet` neural network architecture.
*   `src/dataset.py`: Utilities for downloading and loading MovieLens data.
*   `src/losses.py`: Custom loss functions (BPR, Regression).
*   `src/evaluation.py`: Evaluation metrics (MRR, MSE).
*   `src/utils.py`: Helper functions for tensor manipulation and batching.

