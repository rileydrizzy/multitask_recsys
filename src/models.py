"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=32,
        layer_sizes=None,
        embedding_sharing=True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_sharing = embedding_sharing

        if embedding_sharing:
            self.U, self.Q = self.init_shared_user_and_item_embeddings(
                num_users, num_items, embedding_dim
            )
        else:
            (
                self.U_reg,
                self.Q_reg,
                self.U_fact,
                self.Q_fact,
            ) = self.init_separate_user_and_item_embeddings(
                num_users, num_items, embedding_dim
            )

        self.B = self.init_item_bias(num_users, num_items)

        if layer_sizes is None:
            layer_sizes = [96, 64]
        self.mlp_layers = self.init_mlp_layers(layer_sizes)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Only need to compute values for user and item at the same index.
        For example, interaction and score between (user_ids[1] w.r.t item_ids[1]), ...,\
            (user_ids[batch] w.r.t item_ids[batch])

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        if self.embedding_sharing:
            predictions, score = self.forward_with_embedding_sharing(user_ids, item_ids)
        else:
            predictions, score = self.forward_without_embedding_sharing(
                user_ids, item_ids
            )

        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")

        return predictions, score

    def init_shared_user_and_item_embeddings(self, num_users, num_items, embedding_dim):
        """
        Initializes shared user and item embeddings
        used in both factorization and regression tasks

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.
        embedding_dim: int, optional
            Dimensionality of the latent representations.


        Returns
        -------

        U: ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q: ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        """
        U = Q = None

        U = ScaledEmbedding(num_users, embedding_dim)
        Q = ScaledEmbedding(num_items, embedding_dim)

        return U, Q

    def init_separate_user_and_item_embeddings(
        self, num_users, num_items, embedding_dim
    ):
        """
        Initializes separate user and item embeddings
        where one will be used for factorization (ie _fact) and
        other for regression tasks (ie _reg)

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.
        embedding_dim: int, optional
            Dimensionality of the latent representations.


        Returns
        -------

        U_reg: first ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q_reg: first ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        U_fact: second ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q_fact: second ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)

        Note: Order does matter here! Please declare the layers in the order
        they are returned.
        """
        U_reg = Q_reg = U_fact = Q_fact = None

        U_reg = ScaledEmbedding(num_users, embedding_dim)
        Q_reg = ScaledEmbedding(num_items, embedding_dim)

        U_fact = ScaledEmbedding(num_users, embedding_dim)
        Q_fact = ScaledEmbedding(num_items, embedding_dim)

        return U_reg, Q_reg, U_fact, Q_fact

    def init_item_bias(self, num_users, num_items):
        """
        Initializes item bias terms

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.

        Returns
        -------
        B: ZeroEmbedding layer for items
            nn.Embedding of shape (num_items, 1)
        """
        B = None

        # Item bias terms (Matrix Factorization Only)
        B = ZeroEmbedding(num_items, 1)

        return B

    def init_mlp_layers(self, layer_sizes):
        """
        Initializes MLP layer for regression task

        Parameters
        ----------

        layer_sizes: list
            List of layer sizes to for the regression network.

        Returns
        -------

        mlp_layers: nn.ModuleList
            MLP network containing Linear and ReLU layers
        """
        mlp_layers = None
        # MLP layer for regression task
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], 1))  # Final linear layer
        mlp_layers = nn.Sequential(*layers)
        return mlp_layers

    def forward_with_embedding_sharing(self, user_ids, item_ids):
        """
        Please see forward() docstrings for reference
        """
        predictions = score = None

        users_shared_emb = self.U(user_ids)
        items_shared_emd = self.Q(item_ids)

        # Regression head

        reg_input = torch.cat(
            [users_shared_emb, items_shared_emd, users_shared_emb * items_shared_emd],
            dim=1,
        )
        reg_output = self.mlp_layers(reg_input)
        score = reg_output.squeeze()

        # Matrix Factorization Head
        factorization_output = torch.sum(
            users_shared_emb * items_shared_emd, dim=1, keepdim=True
        )
        predictions = factorization_output + self.B(item_ids).squeeze()
        predictions = predictions.view(-1)
        print(predictions.shape, score.shape)
        return predictions, score

    def forward_without_embedding_sharing(self, user_ids, item_ids):
        """
        Please see forward() docstrings for reference
        """
        predictions = score = None

        users_reg_emb = self.U_reg(user_ids)
        items_reg_emb = self.Q_reg(item_ids)

        users_fact_emb = self.U_fact(user_ids)
        items_fact_emb = self.Q_fact(item_ids)

        # Regression head

        reg_input = torch.cat(
            [users_reg_emb, items_reg_emb, users_reg_emb * items_reg_emb], dim=1
        )

        reg_output = self.mlp_layers(reg_input)
        score = reg_output.squeeze()

        # Matrix Factorization Head
        factorization_output = torch.sum(
            users_fact_emb * items_fact_emb, dim=1, keepdim=True
        )
        predictions = factorization_output + self.B(item_ids).squeeze()
        predictions = predictions.view(-1)

        return predictions, score
