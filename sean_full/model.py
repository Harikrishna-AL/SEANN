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
            
            # else:
            #     if indices_old is not None:
            #         hebbian_score, hebbian_index, hebbian_mask = self.hebbian_update(
            #             x, x1, i, indices_old=indices_old[i], target=target
            #         )
            #     else:
            #         hebbian_score, hebbian_index, hebbian_mask = self.hebbian_update(
            #             x, x1, i, target=target
            #         )

            #     hebbian_scores.append(hebbian_score)
            #     hebbian_masks.append(hebbian_mask)
            #     hebbian_indices.append(hebbian_index)

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
        batch_size = x.size(0)
        
        if indices_old is not None:
            indices_old = indices_old.unsqueeze(0).repeat(batch_size, 1)
            y = y.scatter(1, indices_old, float('-inf'))
        
        _, topk_indices = torch.topk(y, int(self.percent_winner * x_size), dim=1)  # shape: (batch_size, k)

        # Create winner mask
        avg_post_activation = torch.mean(y, dim=0, keepdim=True)
        _, topk_indices_avg = torch.topk(avg_post_activation, int(self.percent_winner * x_size), dim=1)  # shape: (1, k)
        winner_mask = torch.zeros_like(avg_post_activation)
        winner_mask.scatter_(1, topk_indices_avg, 1.0)
        
        indices = torch.arange(x_size)
        indices[topk_indices_avg] = -1
        indices = indices[indices != -1].squeeze(0)

        y = y * winner_mask  
        
        post_T = y.t()
        y_x = torch.mm(post_T, x) / batch_size
        
        y_y_T = torch.mm(post_T, y) / batch_size
        heb_mask = torch.tril(torch.ones(y_y_T.size(), device=y_y_T.device))
        
        y_y_T_lower = y_y_T * heb_mask
        
        lateral_term = torch.mm(y_y_T_lower, heb_param.weight.data)
        
        delta_w = lr  * (y_x - lateral_term)
        
        weights = heb_param.weight.data + delta_w
        
        with torch.no_grad():
            norm = torch.norm(weights, p=2, dim=1, keepdim=True)
            norm = torch.clamp(norm, min=1e-8)
            weights = weights / norm
            
        # if layer_idx == 0:
        #     heb_param.weight.data = weights
        
        scale = torch.zeros_like(gd_layer.weight.data)
        scale[topk_indices] = delta_w[topk_indices]
        scale = (scale - torch.min(scale)) / (
            torch.max(scale) - torch.min(scale) + 1e-8
        )
        
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
