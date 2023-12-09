"""doc
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
            f"Training for {config.epochs} Epochs, Shared Embedding -> {config.shared_embeddings}, Device => {device}"
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

        logger.success(f"Training completed on {config.epochs}")
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
    main(parser.parse_args())
