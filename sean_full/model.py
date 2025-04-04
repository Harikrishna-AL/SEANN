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

    def __init__(self, input_size, output_size, indexes, inhibition_strength=0.01):
        """
        Initializes the network layers, Hebbian parameters, and hooks for gradient freezing.

        Args:
            input_size (int): Size of the input layer.
            output_size (int): Size of the output layer.
            indexes (list): List of neuron indices to freeze during gradient updates.
        """

        super(NN, self).__init__()

        self.k = 5
        self.inhibition_strength = inhibition_strength
        self.percent_winner = 0.5

        self.linear = nn.ModuleList(
            [
                nn.Linear(input_size, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
                nn.Linear(64, output_size),
            ]
        )

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

        if indexes != [[], [], []]:
            self._register_gradient_hooks(self.indexes)

    def forward(self, x, scalers=None, indexes=None, masks=None, indices_old=None, target=None):
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

        if scalers is not None:
            self.update_indexes(scalers)

        hebbian_scores = []
        hebbian_masks = []
        hebbian_indices = []

        for i, layer in enumerate(self.linear):
            x1 = layer(x)
            if i < len(self.linear) - 1:
                x1 = F.relu(x1)
            if masks is not None and i < len(self.linear) - 1:
                x1 = torch.mul(x1, masks[i])
            if i < len(self.linear) - 1:
                if indices_old is not None:
                    hebbian_score, hebbian_index, hebbian_mask = self.hebbian_update(
                        x, x1, i, indices_old=indices_old[i]
                    )
                else:
                    hebbian_score, hebbian_index, hebbian_mask = self.hebbian_update(
                        x, x1, i
                    )

                hebbian_scores.append(hebbian_score)
                hebbian_masks.append(hebbian_mask)
                hebbian_indices.append(hebbian_index)
            
            else:
                if indices_old is not None:
                    hebbian_score, hebbian_index, hebbian_mask = self.hebbian_update(
                        x, x1, i, indices_old=indices_old[i], target=target
                    )
                else:
                    hebbian_score, hebbian_index, hebbian_mask = self.hebbian_update(
                        x, x1, i, target=target
                    )

                hebbian_scores.append(hebbian_score)
                hebbian_masks.append(hebbian_mask)
                hebbian_indices.append(hebbian_index)

            x = x1
        
        x = nn.Softmax(dim=1)(x)

        return x, hebbian_scores, hebbian_indices, hebbian_masks

    def hebb_forward(self, x, indexes=None):
        hebbian_scores = [] 
        hebbian_masks = []

        for i, layer in enumerate(self.hebb_params):
            x1 = layer(x)
            x1 = F.relu(x1)
            if i < len(self.linear) - 1:
                if indexes is not None:
                    hebbian_score, hebbian_mask = self.hebbian_update(
                        x, x1, i, indices_old=indexes[i]
                    )
                else:
                    hebbian_score, hebbian_mask = self.hebbian_update(x, x1, i)
                hebbian_scores.append(hebbian_score)
                hebbian_masks.append(hebbian_mask)
            x = x1

        return x, hebbian_scores, hebbian_masks

    def hebbian_update(self, x, y, layer_idx, lr=1, threshold=0.5, indices_old=None, target=None):
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

        # heb_param = self.hebb_params[layer_idx]
        heb_param = self.linear[layer_idx]
        gd_layer = self.linear[layer_idx]

        x_size = self.hidden_size_array[layer_idx]

        if target is not None:
            error = target - y
            error = torch.mean(error, dim=0, keepdim=True)

        inhibit_y = y - self.inhibition_strength * (y.sum(dim=1, keepdim=True) - y)
        y = torch.clamp(inhibit_y, min=0)

        theta = torch.mean(y**2, dim=0, keepdim=True)

        if target is not None:
            outer_product = torch.mul(y.unsqueeze(2), x.unsqueeze(1)) + torch.mul(error.unsqueeze(2), x.unsqueeze(1))
        else:
            outer_product = torch.mul(y.unsqueeze(2), x.unsqueeze(1))


        delta_w = lr * (
            torch.sum(outer_product, dim=0) - heb_param.weight.data * theta.T
        )

        delta_w = (delta_w - torch.min(delta_w)) / (torch.max(delta_w) - torch.min(delta_w) + 1e-8)

        y_mean = torch.mean(y, dim=0)

        sort = torch.argsort(y_mean, descending=True)
        if indices_old is not None:
            # remove the neurons that were selected in the previous iteration
            y_mean[indices_old] = -1 * torch.inf
            sort = torch.argsort(y_mean, descending=True)
            winner_idx = sort[: int(x_size * self.percent_winner)]
        else:
            winner_idx = sort[: int(x_size * self.percent_winner)]

        winner_mask = torch.zeros_like(y_mean)
        winner_mask[winner_idx] = 1

        # update the weights of the gd layer by moving the weights in the direction of the hebbian weights
        scale = torch.zeros_like(gd_layer.weight.data)
        scale[winner_idx] = delta_w[winner_idx]
        # scale values between 0 and 1
        scale = (scale - torch.min(scale)) / (
            torch.max(scale) - torch.min(scale) + 1e-8
        )

        indices = sort[int(x_size * self.percent_winner) : -1]

        return scale, indices, winner_mask

    def scale_grad(self, scalers):
        """
        Scales gradients for neurons specified by indexes.

        Args:
            indexes (list): List of neuron indices to scale during gradient updates.

        Returns:
            function: Hook function for modifying gradients during backpropagation.
        """

        def hook(grad):
            if len(scalers) > 0:
                grad *= scalers
            return grad

        return hook

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
        for i, layer in enumerate(self.linear):
            # Check if the layer already has hooks registered and clear them if they exist
            if layer.weight._backward_hooks is not None:
                layer.weight._backward_hooks.clear()
            layer.weight.register_hook(self.scale_grad(indexes[i]))

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
