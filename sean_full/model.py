import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import get_data_separate
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


class NN(nn.Module):
    """
    Neural network class with Hebbian learning mechanisms.
    """

    def __init__(self, input_size, output_size, indexes):
        """
        Initializes the network layers, Hebbian parameters, and hooks for gradient freezing.

        Args:
            input_size (int): Size of the input layer.
            output_size (int): Size of the output layer.
            indexes (list): List of neuron indices to freeze during gradient updates.
        """

        super(NN, self).__init__()

        self.k = 5

        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, output_size)

        # Define the Hebbian parameters corresponding to each layer
        self.hebb_params = nn.ModuleList(
            [
                nn.Linear(input_size, 256, bias=False),
                nn.Linear(256, 128, bias=False),
                nn.Linear(128, 64, bias=False),
                nn.Linear(64, output_size, bias=False),
            ]
        )

        for i, heb_param in enumerate(self.hebb_params):
            nn.init.kaiming_normal_(heb_param.weight)
            heb_param.weight.requires_grad = False

        self.indexes = indexes
        self.hidden_size_array = [256, 128, 64, output_size]

        self._register_gradient_hooks(self.indexes)

    def forward(self, x, indexes=None, masks=None):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.
            indexes (list, optional): New indexes to update for freezing gradients.
            masks (list, optional): Masking values applied to specific layers during forward pass.

        Returns:
            torch.Tensor: Output tensor.
            list: Hebbian scores for each layer.
            list: Hebbian masks for each layer.
        """

        if indexes is not None:
            self.update_indexes(indexes)

        hebbian_scores = []
        hebbian_masks = []
        layers = [self.linear1, self.linear2, self.linear3, self.linear4]

        for i, layer in enumerate(layers):
            x1 = layer(x)
            x1 = F.relu(x1)
            if masks is not None and i < len(layers) - 1:
                x1 = torch.mul(x1, masks[i])
            if i < len(layers) - 1:
                hebbian_score, hebbian_mask = self.hebbian_update(x, i)
                hebbian_scores.append(hebbian_score)
                hebbian_masks.append(hebbian_mask)
            x = x1

        return x, hebbian_scores, hebbian_masks

    def hebbian_update(self, x, layer_idx, lr=1, threshold=0.5):
        """
        Updates Hebbian parameters based on Hebbian learning principles.

        Args:
            x (torch.Tensor): Input tensor.
            layer_idx (int): Index of the current layer in the Hebbian parameter list.
            lr (float, optional): Learning rate for Hebbian updates.
            threshold (float, optional): Threshold for Hebbian score masking.

        Returns:
            torch.Tensor: Indices of neurons with Hebbian scores below the threshold.
            torch.Tensor: Mask tensor for the layer based on Hebbian scores.
        """

        heb_param = self.hebb_params[layer_idx]
        x_size = self.hidden_size_array[layer_idx]

        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        y = heb_param(x_norm)
        y_norm = y / torch.norm(y, dim=1, keepdim=True)
        winner_idx = torch.argsort(y_norm)[-self.k :]

        theta = torch.mean(y_norm**2, dim=0, keepdim=True)
        outer_product = torch.mul(y_norm.unsqueeze(2), x_norm.unsqueeze(1))
        heb_param.weight.data[winner_idx] += lr * (
            torch.sum(outer_product, dim=0)[winner_idx]
            - heb_param.weight.data[winner_idx] * theta.T[winner_idx]
        )

        # Calculate Hebbian scores and masks
        hebbian_score = torch.sum(heb_param.weight.data, dim=1)
        hebbian_score = (hebbian_score - torch.min(hebbian_score)) / (
            torch.max(hebbian_score) - torch.min(hebbian_score) + 1e-8
        )
        hebbian_score_indices = torch.where(hebbian_score < threshold)[0]
        hebbian_mask = torch.ones_like(hebbian_score)
        hebbian_mask[hebbian_score_indices] = 0
        return hebbian_score_indices, hebbian_mask

    def freeze_grad(self, indexes):
        """
        Freezes gradients for neurons specified by indexes.

        Args:
            indexes (list): List of neuron indices to freeze during gradient updates.

        Returns:
            function: Hook function for modifying gradients during backpropagation.
        """

        def hook(grad):
            if len(indexes) > 0:
                indexes_arr = (
                    indexes.cpu().numpy()
                    if isinstance(indexes, torch.Tensor)
                    else indexes
                )
                grad[indexes_arr] = 0
            return grad

        return hook

    def _register_gradient_hooks(self, indexes):
        """
        Registers hooks for freezing gradients on specified neurons.

        Args:
            indexes (list): List of neuron indices to freeze during gradient updates.
        """

        layers = [self.linear1, self.linear2, self.linear3]
        for i, layer in enumerate(layers):
            # Check if the layer already has hooks registered and clear them if they exist
            if layer.weight._backward_hooks is not None:
                layer.weight._backward_hooks.clear()
            layer.weight.register_hook(self.freeze_grad(indexes[i]))

    def update_indexes(self, new_indexes):
        """
        Updates the indexes of neurons for freezing and re-registers gradient hooks.

        Args:
            new_indexes (list): New list of neuron indexes to freeze.
        """

        self.indexes = new_indexes
        self._register_gradient_hooks(new_indexes)

    def reinitialize_hebbian_parameters(self, init_type="zero"):
        """
        Reinitializes the Hebbian parameters.

        Args:
            init_type (str, optional): Initialization type ('zero' or 'normal').
        """

        for param in self.hebb_params.parameters():
            if init_type == "zero":
                nn.init.constant_(param, 0)
            elif init_type == "normal":
                nn.init.kaiming_normal_(param)
