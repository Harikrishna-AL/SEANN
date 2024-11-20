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

        self.linear = nn.ModuleList(
            [
                nn.Linear(input_size, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 64),
                nn.Linear(64, output_size),
            ]
        )
        # self.linear1 = nn.Linear(input_size, 256)
        # self.linear2 = nn.Linear(256, 128)
        # self.linear3 = nn.Linear(128, 64)
        # self.linear4 = nn.Linear(64, output_size)

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

        if indexes != [[],[],[]]:
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

        if indexes is not None and indexes != [[],[],[]]:
            self.update_indexes(indexes)

        # layers = [self.linear1, self.linear2, self.linear3, self.linear4]

        for i, layer in enumerate(self.linear):
            x1 = layer(x)
            x1 = F.relu(x1)
            if masks is not None and i < len(self.linear) - 1:
                x1 = torch.mul(x1, masks[i])
            # if i < len(self.linear) - 1:
            #     hebbian_score, hebbian_mask = self.hebbian_update(x, i)
            #     hebbian_scores.append(hebbian_score)
            #     hebbian_masks.append(hebbian_mask)
            x = x1

        return x
    
    def hebb_forward(self, x, indexes=None):
        hebbian_scores = []
        hebbian_masks = []
        for i, layer in enumerate(self.hebb_params):
            x1 = layer(x)
            x1 = F.relu(x1)
            if i < len(self.linear) - 1:
                if indexes is not None:
                    hebbian_score, hebbian_mask = self.hebbian_update(x, x1, i, indices_old=indexes[i])
                else:
                    hebbian_score, hebbian_mask = self.hebbian_update(x, x1, i)
                hebbian_scores.append(hebbian_score)
                hebbian_masks.append(hebbian_mask)
            x = x1

        return x, hebbian_scores, hebbian_masks

    def hebbian_update(self, x, y, layer_idx, lr=1, threshold=0.5, indices_old = None):
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
        gd_layer = self.linear[layer_idx]

        x_size = self.hidden_size_array[layer_idx]

        # y = heb_param(x)
        # y_norm = y / torch.norm(y, dim=1, keepdim=True)
        # winner_idx = torch.argsort(y_norm)[-self.k :]
        inhibit_y = y - self.inhibition_strength * (y.sum(dim=1, keepdim=True) - y)
        y = torch.clamp(inhibit_y, min=0)

        theta = torch.mean(y**2, dim=0, keepdim=True)
        outer_product = torch.mul(y.unsqueeze(2), x.unsqueeze(1))
        heb_param.weight.data += lr * (
            torch.sum(outer_product, dim=0)
            - heb_param.weight.data * theta.T
        )
        #normalize the weights
        heb_param.weight.data = heb_param.weight.data / torch.norm(heb_param.weight.data, dim=1, keepdim=True)
        gd_layer.weight.data = gd_layer.weight.data / torch.norm(gd_layer.weight.data, dim=1, keepdim=True)

        # select top k% of neurons
        y_mean = torch.mean(y, dim=0)

        sort = torch.argsort(y_mean, descending=True)
        if indices_old is not None:
            # remove the neurons that were selected in the previous iteration
            y_mean[indices_old] = -1*torch.inf
            winner_idx = sort[:int(x_size * 0.4)]
        else:
            winner_idx = sort[:int(x_size * 0.4)]
            
        winner_mask = torch.zeros_like(y_mean)
        winner_mask[winner_idx] = 1

        # Calculate Hebbian scores and masks
        # hebbian_score = torch.sum(heb_param.weight.data, dim=1)
        # hebbian_score = (hebbian_score - torch.min(hebbian_score)) / (
        #     torch.max(hebbian_score) - torch.min(hebbian_score) + 1e-8
        # )
        # # print(hebbian_score)
        # hebbian_score_indices = torch.where(hebbian_score < threshold)[0]
        # hebbian_mask = torch.ones_like(hebbian_score)
        # hebbian_mask[hebbian_score_indices] = 0

        
        #update the weights of the gd layer by moving the weights in the direction of the hebbian weights
        delta_w = torch.zeros_like(gd_layer.weight.data)
        delta_w[winner_idx] = heb_param.weight.data[winner_idx]
        # take moving average of the weights
        # gd_layer.weight.data = 0.9 * gd_layer.weight.data + 0.1 * delta_w

        dot_product = torch.dot(gd_layer.weight.data.flatten(), delta_w.flatten())
        bp_norm_sqr = torch.linalg.norm(gd_layer.weight.data)**2
        if bp_norm_sqr != 0:
            projection = (dot_product / bp_norm_sqr) * gd_layer.weight.data
            hebb_orthogonal = delta_w - projection
            gd_layer.weight.data += hebb_orthogonal

        indices = sort[int(x_size * 0.4):-1]

        return indices, winner_mask

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
        
        # layers = [self.linear1, self.linear2, self.linear3]
        for i, layer in enumerate(self.linear[:-1]):
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
