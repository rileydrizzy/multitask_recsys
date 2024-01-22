""" 
This script defines the main function for training a MultiTaskNet model on the MovieLens dataset.
The MovieLens dataset is used to create a Multi-Task Neural Network, consisting of both 
factorization and regression tasks.

Modules Imported:
- dataset: Provides functions for obtaining the MovieLens dataset.
- evaluation: Contains metrics like MRR (Mean Reciprocal Rank) and MSE (Mean Squared Error).
- models: Defines the MultiTaskNet model.
- multitask: Implements the MultitaskModel for handling both factorization and regression tasks.

Usage:
- Ensure the required dependencies are installed.
- Adjust configuration options in the command-line arguments or within the script.
- Run the script to train the MultiTaskNet model on the MovieLens dataset.

Command-line Arguments:
- logdir: Path to the directory for storing TensorBoard logs.
- test_fraction: Fraction of the dataset to use for testing.
- shared_embeddings: Option to use shared embeddings in the MultiTaskNet.
- factorization_weight: Weight for the factorization task.
- regression_weight: Weight for the regression task.
- gpu: Flag indicating whether to use GPU for training.
- epochs: Number of training epochs.

Example Usage:
```bash
python script_name.py --logdir logs/ --test_fraction 0.2 --shared_embeddings --factorization_weight 0.5 --regression_weight 0.5 --gpu --epochs 10

"""


import argparse
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import get_movielens_dataset
from evaluation import mrr_score, mse_score
from models import MultiTaskNet
from multitask import MultitaskModel


def main(config):
    """
    Main function for training a MultiTaskNet model on the MovieLens dataset.

    Parameters
    ----------
    config : argparse.Namespace
        Configuration options.
    """

    try:
        print(config)
        writer = SummaryWriter(config.logdir)

        logger.info("Getting the dataset")
        dataset = get_movielens_dataset(variant="100K")
        train, test = dataset.random_train_test_split(
            test_fraction=config.test_fraction
        )

        net = MultiTaskNet(
            train.num_users, train.num_items, embedding_sharing=config.shared_embeddings
        )
        model = MultitaskModel(
            interactions=train,
            representation=net,
            factorization_weight=config.factorization_weight,
            regression_weight=config.regression_weight,
            use_cuda=config.gpu,
        )
        if config.gpu:
            device = "CUDA"
        else:
            device = "CPU"

        logger.info(
            f"Training for {config.epochs} Epochs, Shared Embedding => {config.shared_embeddings}, Device => {device}"
        )
        for epoch in tqdm(range(config.epochs)):
            factorization_loss, score_loss, joint_loss = model.fit(train)
            mrr = mrr_score(model, test, train)
            mse = mse_score(model, test)
            writer.add_scalar("training/Factorization Loss", factorization_loss, epoch)
            writer.add_scalar("training/MSE", score_loss, epoch)
            writer.add_scalar("training/Joint Loss", joint_loss, epoch)
            writer.add_scalar("eval/Mean Reciprocal Rank", mrr, epoch)
            writer.add_scalar("eval/MSE", mse, epoch)

        logger.success(f"Training completed on {config.epochs} Epochs")
    except Exception as error:
        logger.exception(f"Training failed due to {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_fraction", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--factorization_weight", type=float, default=0.995)
    parser.add_argument("--regression_weight", type=float, default=0.005)
    parser.add_argument("--shared_embeddings", default=True, action="store_true")
    parser.add_argument(
        "--no_shared_embeddings", dest="shared_embeddings", action="store_false"
    )
    parser.add_argument("--logdir", type=str, default="run/shared=True_LF=0.99_LR=0.01")
    parser.add_argument("--gpu", type=bool, default=False)
    config_ = parser.parse_args()
    main(config_)
