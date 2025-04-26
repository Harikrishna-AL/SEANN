import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import get_data_separate
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


class RNNGate(nn.Module):
    """
    RNN gate class for controlling the flow of information in the network.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN gate parameters.

        Args:
            input_size (int): Size of the input layer.
            hidden_size (int): Size of the hidden layer.
        """
        super(RNNGate, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the RNN gate.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the RNN and linear layer.
        """
        out, _ = self.rnn(x)
        out = self.linear(out)
        return out

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
        self.percent_winner = 0.2

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

    def forward(self, x, scalers=None, indexes=None, masks=None, indices_old=None, target=None, selection_method="hebbian"):
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
        common_indices = []

        for i, layer in enumerate(self.linear):
            x1 = layer(x)
            is_final_layer = (i == len(self.linear) - 1)

            if masks is not None: # check later why multiplying the mask of the last layer as well causes a drop is accuracy values.
                x1 = torch.mul(x1, masks[i])
                
            x1 = F.relu(x1) if not is_final_layer else x1

            if selection_method == "hebbian":
                hebbian_score, hebbian_index, hebbian_mask, common_index = self.hebbian_update(
                    x, x1, i, indices_old=indices_old[i], target=target if is_final_layer else None
                )
            elif selection_method == "random":
                hebbian_score, hebbian_index, hebbian_mask, common_index = self.random_selection(
                    x, x1, i, indices_old=indices_old[i], target=target if is_final_layer else None
                )
            else:
                raise ValueError("Invalid selection method. Choose 'hebbian' or 'random'.")

            hebbian_scores.append(hebbian_score)
            hebbian_masks.append(hebbian_mask)
            hebbian_indices.append(hebbian_index)
            if common_index is not None:
                common_indices.append(common_index)            

            x = x1
        
        x = nn.Softmax(dim=1)(x)

        return x, hebbian_scores, hebbian_indices, hebbian_masks, common_indices

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

    def hebbian_update(self, x, y, layer_idx, lr=0.00005, threshold=0.5, indices_old=None, target=None):
        """
        Calculates Hebbian-derived scores, masks, and scales for a layer.
        Handles final layer differently using the one-hot target.
        """

        gd_layer = self.linear[layer_idx]
        x_size = self.hidden_size_array[layer_idx] # Size of the output dimension of the layer

        batch_size = x.size(0)
        common_indices = None

        is_final_layer = (target is not None) # Assuming target is only non-None for the final layer

        if not is_final_layer:
            post_T = y.t() # Shape: (output_size, batch_size)
            pre = x        # Shape: (batch_size, input_size)

            y_x = torch.mm(post_T, pre) / batch_size # Shape: (output_size, input_size) - Pre-post correlation average

            y_y_T = torch.mm(post_T, y) / batch_size # Shape: (output_size, output_size) - Post-post correlation average
            heb_mask_tril = torch.tril(torch.ones(y_y_T.size(), device=y_y_T.device))
            y_y_T_lower = y_y_T * heb_mask_tril

            lateral_term = torch.mm(y_y_T_lower, gd_layer.weight.data) # Shape: (output_size, input_size)

            delta_w = lr * (y_x - lateral_term)

            modified_weights = gd_layer.weight.data + delta_w
            with torch.no_grad():
                 norm = torch.norm(modified_weights, p=2, dim=1, keepdim=True) # Shape: (output_size, 1)
                 norm = torch.clamp(norm, min=1e-8) # Avoid division by zero
                 normalized_modified_weights = modified_weights / norm

            # gd_layer.weight.data = normalized_modified_weights # Uncomment this line if you *do* intend this direct update + normalization

            hebbian_scores = torch.norm(normalized_modified_weights.detach(), p=2, dim=1) # Shape: (output_size)

            # if indices_old is not None:
            #      # scatter expects index to be long tensor
            #      hebbian_scores = hebbian_scores.scatter(0, indices_old.long(), float('-inf'))

            num_winners = int(self.percent_winner * x_size)
            if num_winners == 0 and x_size > 0: num_winners = 1 # Ensure at least one winner if layer exists
            elif x_size == 0: num_winners = 0 # Handle empty layer gracefully

            if num_winners > 0:
                _, topk_indices_hebbian = torch.topk(hebbian_scores, num_winners) # Shape: (num_winners)
                if indices_old is not None:
                    # all_indices = torch.arange(x_size, device=y.device)
                    # is_allowed = ~torch.isin(all_indices, indices_old.long())
                    # allowed_indices = all_indices[is_allowed]
                    # indices_old = allowed_indices
                    common_indices = torch.isin(topk_indices_hebbian, indices_old.long())
                    common_indices = topk_indices_hebbian[common_indices]
                    if len(common_indices) > int(0.5 * num_winners):
                        hebbian_scores = hebbian_scores.scatter(0, common_indices[:int(0.5*num_winners)], float('-inf'))
                        _, topk_indices_hebbian = torch.topk(hebbian_scores, num_winners, largest=True, sorted=False)
                    else:
                        hebbian_scores = hebbian_scores.scatter(0, common_indices, float('-inf'))
                        _, topk_indices_hebbian = torch.topk(hebbian_scores, num_winners, largest=True, sorted=False)
                        
                         
            else:
                topk_indices_hebbian = torch.tensor([], dtype=torch.long, device=y.device)


            hebbian_mask = torch.zeros(1, x_size, device=y.device)
            if num_winners > 0:
                 hebbian_mask.scatter_(1, topk_indices_hebbian.unsqueeze(0), 1.0) # Indices need shape (1, num_winners)

            all_indices = torch.arange(x_size, device=y.device)
            indices_non_winners = all_indices[hebbian_mask.squeeze(0) == 0] # Select indices where mask is 0

            scale = torch.zeros_like(gd_layer.weight.data) # Shape: (output_size, input_size)

            if num_winners > 0:
                 scale[topk_indices_hebbian] = normalized_modified_weights[topk_indices_hebbian]

            return scale, indices_non_winners, hebbian_mask, common_indices


        else:
            x_size = self.hidden_size_array[layer_idx] # Size of the output dimension of the layer
            if target is None:
                 print("Warning: Target is None for final layer Hebbian update.")
                 scale_output = torch.zeros_like(gd_layer.weight.data)
                 hebbian_mask = torch.ones(1, x_size, device=y.device)
                 indices_non_winners = torch.tensor([], dtype=torch.long, device=y.device)
                 return scale_output, indices_non_winners, hebbian_mask

            target_onehot = target
            post_T_supervised = target_onehot.t() # Shape: (output_size, batch_size)
            pre_supervised = x                     # Shape: (batch_size, input_size)

            correlation_term = torch.mm(post_T_supervised, pre_supervised) / batch_size # Shape: (output_size, input_size)

            scale_output = correlation_term.detach() # Shape: (output_size, input_size)
            min_scale = torch.min(scale_output)
            max_scale = torch.max(scale_output)
            if max_scale - min_scale > 1e-8:
                 scale_output = (scale_output - min_scale) / (max_scale - min_scale)
          
            hebbian_scores = torch.norm(scale_output, p=2, dim=1) # Shape: (output_size)
            if indices_old is not None:
                hebbian_scores = hebbian_scores.scatter(0, indices_old.long(), float('-inf'))
            _, topk_indices_hebbian = torch.topk(hebbian_scores, int(self.percent_winner * x_size)) # Shape: (1)
            hebbian_mask = torch.zeros(1, x_size, device=y.device)
            hebbian_mask.scatter_(1, topk_indices_hebbian.unsqueeze(0), 1.0)
            indices_non_winners = torch.arange(x_size, device=y.device)[hebbian_mask.squeeze(0) == 0] # Select indices where mask is 0

            return scale_output, indices_non_winners, hebbian_mask, common_indices
        
    def random_selection(self, x, y, layer_idx, indices_old=None, target=None):
        """
        Selects neurons randomly for scaling/masking.
        Handles final layer similarly regarding num_winners based on percent_winner.
        Ignores Hebbian scores.
        """
        gd_layer = self.linear[layer_idx]
        x_size = self.hidden_size_array[layer_idx] # Size of the output dimension of the layer
        batch_size = x.size(0)
        is_final_layer = (target is not None)

        # Calculate the number of winners based on percent_winner, same as Hebbian
        num_winners = int(self.percent_winner * x_size)
        if num_winners == 0 and x_size > 0: num_winners = 1 # Ensure at least one winner if layer exists
        elif x_size == 0: num_winners = 0 # Handle empty layer gracefully

        # Get all possible indices for this layer
        all_indices = torch.arange(x_size, device=y.device)

        if indices_old is not None and indices_old.numel() > 0:
            # Create a boolean mask where True means the index is NOT in indices_old
            is_allowed = ~torch.isin(all_indices, indices_old.long())
            allowed_indices = all_indices[is_allowed]
        else:
            # If no old indices, all indices are allowed
            allowed_indices = all_indices

        num_candidates = allowed_indices.size(0)
        # Ensure we don't try to select more winners than available candidates
        num_winners = min(num_winners, num_candidates)


        # --- Random Selection ---
        if num_winners > 0:
            # Randomly permute the allowed indices and take the first 'num_winners'
            perm = torch.randperm(num_candidates, device=y.device)
            topk_indices_random = allowed_indices[perm[:num_winners]] # Shape: (num_winners)
        else:
            topk_indices_random = torch.tensor([], dtype=torch.long, device=y.device)

        random_mask = torch.zeros(1, x_size, device=y.device)
        if num_winners > 0:
            random_mask.scatter_(1, topk_indices_random.unsqueeze(0), 1.0)

        # Get the indices of the non-selected neurons (complement of random winners)
        indices_non_winners = all_indices[random_mask.squeeze(0) == 0]

        if not is_final_layer:
            # Use raw y for correlation
            post_T = y.t() # Shape: (output_size, batch_size)
            pre = x        # Shape: (batch_size, input_size)
            correlation_term = torch.mm(post_T, pre) / batch_size # Shape: (output_size, input_size)
            with torch.no_grad():
                norm_correlation = torch.norm(correlation_term.detach(), p=2, dim=1, keepdim=True) # Shape: (output_size, 1)
                norm_correlation = torch.clamp(norm_correlation, min=1e-8) # Avoid division by zero
                normalized_correlation = correlation_term.detach() / norm_correlation # Shape: (output_size, input_size)

        else: # Final layer (supervised)
            if target is None:
                print("Warning: Target is None for final layer random selection scale calculation.")
                # Return default zero scale, all-one mask, and empty non-winners
                scale_output = torch.zeros_like(gd_layer.weight.data)
                random_mask_output = torch.ones(1, x_size, device=y.device) # Default to all active if no target for final layer
                indices_non_winners_output = torch.tensor([], dtype=torch.long, device=y.device)
                return scale_output, indices_non_winners_output, random_mask_output

            # Use one-hot target for correlation calculation for final layer scale
            post_T_supervised = target.t() # Shape: (output_size, batch_size)
            pre_supervised = x
            correlation_term = torch.mm(post_T_supervised, pre_supervised) / batch_size # Shape: (output_size, input_size)
            with torch.no_grad():
                norm_correlation = torch.norm(correlation_term.detach(), p=2, dim=1, keepdim=True) # Shape: (output_size, 1)
                norm_correlation = torch.clamp(norm_correlation, min=1e-8) # Avoid division by zero
                normalized_correlation = correlation_term.detach() / norm_correlation # Shape: (output_size, input_size)

         # Shape: (output_size, input_size)
        if num_winners > 0:
            # scale[topk_indices_random] = normalized_correlation[topk_indices_random]
            scale = torch.zeros_like(gd_layer.weight.data)
            scale[topk_indices_random] = 1.0
        
        if is_final_layer:
            scale = torch.zeros_like(gd_layer.weight.data)
            scale[topk_indices_random] = normalized_correlation[topk_indices_random]
            hebbian_scores = torch.norm(scale, p=2, dim=1) # Shape: (output_size)
            if indices_old is not None:
                hebbian_scores = hebbian_scores.scatter(0, indices_old.long(), float('-inf'))
            _, topk_indices_hebbian = torch.topk(hebbian_scores, int(self.percent_winner * x_size)) # Shape: (1)
            hebbian_mask = torch.zeros(1, x_size, device=y.device)
            hebbian_mask.scatter_(1, topk_indices_hebbian.unsqueeze(0), 1.0)
            random_mask = hebbian_mask
            indices_non_winners = torch.arange(x_size, device=y.device)[hebbian_mask.squeeze(0) == 0]

        return scale, indices_non_winners, random_mask

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
