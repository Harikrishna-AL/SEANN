import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import get_data_separate
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

class NN(nn.Module):
    def __init__(self, input_size, output_size, indexes):
        super(NN, self).__init__()
 
        self.linear1 = nn.Linear(input_size, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, output_size)

        # Define the Hebbian parameters corresponding to each layer
        self.hebb_params = nn.ModuleList([
            nn.Linear(input_size, 256, bias=False),
            nn.Linear(256, 128, bias=False),
            nn.Linear(128, 64, bias=False),
            nn.Linear(64, output_size, bias=False)
        ])

        for i, heb_param in enumerate(self.hebb_params):
            nn.init.normal_(heb_param.weight, 0, 0.01)
            heb_param.weight.requires_grad = False
        
        self.indexes = indexes
        self.hidden_size_array = [256, 128, 64, output_size]

        self._register_gradient_hooks(self.indexes)

    def _register_gradient_hooks(self, indexes):
        layers = [self.linear1, self.linear2, self.linear3]
        for i, layer in enumerate(layers):
            # Check if the layer already has hooks registered and clear them if they exist
            if layer.weight._backward_hooks is not None:
                layer.weight._backward_hooks.clear()
            layer.weight.register_hook(self.freeze_grad(indexes[i]))


    def forward(self, x, indexes=None, masks=None):
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
        heb_param = self.hebb_params[layer_idx]
        x_size = self.hidden_size_array[layer_idx]

        # Normalize the inputs and outputs
        x_norm = x / torch.norm(x, dim=1, keepdim=True)
        y = heb_param(x_norm)
        y_norm = y / torch.norm(y, dim=1, keepdim=True)
        # y_norm = y
        # x_norm = x
        
        # Calculate the Hebbian weight update
        # print("x_norm shape: ", x_norm.shape)
        # print("y_norm shape: ", y_norm.shape)
        theta = torch.mean(y_norm**2)

        outer_product = torch.mul(y_norm.unsqueeze(2), x_norm.unsqueeze(1)) 
        # second_term = torch.mul(heb_param.weight.data, y_norm.unsqueeze(2))
        # print("second term shape: ", second_term.shape)
        # print("outer product shape: ", outer_product.shape)
        heb_param.weight.data += lr * (torch.sum(outer_product, dim=0) - theta * heb_param.weight.data)
        
        # Calculate Hebbian scores and masks
        hebbian_score = torch.sum(heb_param.weight.data, dim=1)
        hebbian_score = (hebbian_score - torch.min(hebbian_score)) / (torch.max(hebbian_score) - torch.min(hebbian_score) + 1e-8) 
        # print("Hebbian score: ", hebbian_score)
        hebbian_score_indices = torch.where(hebbian_score < threshold)[0]
        hebbian_mask = torch.ones_like(hebbian_score)
        hebbian_mask[hebbian_score_indices] = 0
        # print("Hebbian score: ", hebbian_score)
    
        return hebbian_score_indices, hebbian_mask

    def freeze_grad(self, indexes):
        def hook(grad):
            if len(indexes) > 0:
                indexes_arr = indexes.cpu().numpy() if isinstance(indexes, torch.Tensor) else indexes
                grad[indexes_arr] = 0
            return grad
        return hook

    def update_indexes(self, new_indexes):
        self.indexes = new_indexes
        self._register_gradient_hooks(new_indexes)

    def reinitialize_hebbian_parameters(self, init_type='zero'):
        for param in self.hebb_params.parameters():
            if init_type == 'zero':
                nn.init.constant_(param, 0)
            elif init_type == 'normal':
                nn.init.normal_(param, 0, 0.01)

def forwardprop_and_backprop(model,lr, data_loader, continual=False, list_of_indexes=None, masks=None, scheduler=None, optimizer=None):
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    loss_total = 0
    model.train()
    # target = torch.randint(0, 10, (64,))
    for i, (data, target) in enumerate(tqdm(data_loader)):
        optimizer.zero_grad()
        data = data.view(-1, 784)
        data, target = data.to(device), target.to(device)
        if not continual:
            output, list_of_indexes_out, masks = model(data, indexes=list_of_indexes, masks=masks)
            if i == len(data_loader) - 1:
                list_of_indexes = list_of_indexes_out
            
        else:
            output, list_of_indexes_out, masks_out = model(data, indexes=list_of_indexes, masks=masks)
        
        loss = criterion(output, target)
        loss.backward()
        # print(model.linear4.weight.grad)
        optimizer.step()
        loss_total += loss.item()

    scheduler.step()

    print("Avg loss: ", loss_total/len(data_loader))
    # print("mask: ", masks)
    return list_of_indexes, masks, model, optimizer

def calc_percentage_of_zero_grad(masks):
    total = 0
    zero = 0
    for mask in masks:
        total += mask.numel()
        zero += torch.sum(mask == 0).item()
    return  (1 - zero/total)*100
# Example usage
# seed = torch.randint(0, 1000, (1,)).item()
seed = 760 #verified
print("Seed: ", seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


data_loader_1, data_loader_2, test_loader_1, test_loader_2 = get_data_separate(batch_size=64)
# data = torch.randn(64, 784)
list_of_indexes = [[], [], []]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
masks = [torch.ones(256).to(device), torch.ones(128).to(device), torch.ones(64).to(device)]

# masks = [torch.ones(512).to(device=),torch.ones(256), torch.ones(128), torch.ones(64)]


original_model = NN(784, 10, indexes=list_of_indexes).to(device)
optimizer = optim.SGD(original_model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1) 
for i in range(10):
    task1_indices, task1_masks, task1_model, optimizer = forwardprop_and_backprop(original_model,0.01, data_loader_1, list_of_indexes=list_of_indexes, masks=masks, optimizer=optimizer, scheduler=scheduler)
    list_of_indexes = task1_indices
    # print("percentage of zero gradients: ",calc_percentage_of_zero_grad(original_model))

indices = []
new_masks = []
layer_sizes = [256, 128, 64]
for i in range(len(layer_sizes)):
    indices.append(torch.tensor([j for j in range(layer_sizes[i]) if j not in task1_indices[i]]).to(device))
    mask = torch.tensor([1 if k in task1_indices[i] else 0 for k in range(layer_sizes[i])]).to(device)
    new_masks.append(mask)
print("Task 1 indices: ", task1_indices)
print("Task 1 masks: ", task1_masks)
print("indices: ", indices)
print("masks: ", new_masks)
print("Percentage of frozen neurons: ", calc_percentage_of_zero_grad(task1_masks))


# correct = 0
# accuracies = []
# original_model.eval()
# print("### Testing Task 1###")
# for data, target in test_loader_1:
#     data = data.view(-1, 784)
#     data, target = data.to(device), target.to(device)
#     output, indices, masks = task1_model(data, masks=task1_masks)
#     # check the accuracy
#     predicted = output.argmax(dim=1, keepdim=True)
#     correct += predicted.eq(target.view_as(predicted)).sum().item()

# print(f"Accuracy for Task 1 before Task 2: {100* correct/len(test_loader_1.dataset)}%")
# accuracies.append(100* correct/len(test_loader_1.dataset))

task1_model.reinitialize_hebbian_parameters(init_type='normal')

print("### Task 2 ###")
for i in range(10):
        task2_indices, task2_masks, task2_model, optimizer = forwardprop_and_backprop(task1_model,0.1, data_loader_2, list_of_indexes=indices, masks=new_masks, continual=True, optimizer=None, scheduler=scheduler)
    # print("Percentage of frozen neurons: ", calc_percentage_of_zero_grad(task2_masks))
    # print("percentage of zero gradients: ",calc_percentage_of_zero_grad(original_model))

#test
# forwardprop_and_backprop(original_model, test_loader, list_of_indexes=list_of_indexes)
correct = 0
accuracies = []
original_model.eval()
print("### Testing Task 1###")
for data, target in test_loader_1:
    data = data.view(-1, 784)
    data, target = data.to(device), target.to(device)
    output, indices, masks = task2_model(data, masks=task1_masks)
    # check the accuracy
    predicted = output.argmax(dim=1, keepdim=True)
    correct += predicted.eq(target.view_as(predicted)).sum().item()

print(f"Accuracy for Task 1: {100* correct/len(test_loader_1.dataset)}%")
accuracies.append(100* correct/len(test_loader_1.dataset))

correct = 0
print("### Testing Task 2###")
for data, target in test_loader_2:
    data = data.view(-1, 784)
    data, target = data.to(device), target.to(device)
    output, indices, masks = task2_model(data, masks=task2_masks)
    # check the accuracy
    predicted = output.argmax(dim=1, keepdim=True)
    correct += predicted.eq(target.view_as(predicted)).sum().item()

print(f"Accuracy for Task 2: {100* correct/len(test_loader_2.dataset)}%")
accuracies.append(100* correct/len(test_loader_2.dataset))

forgetting = 0





