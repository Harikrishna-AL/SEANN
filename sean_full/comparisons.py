from data import get_data_separate_domain_incremental, get_data_separate_dynamic


def train_and_evaluate(model, train_loaders, test_loaders, learning_scenario="domain", baseline="ewc"):
    """
    Train and evaluate a model on domain or task incremental learning using a specified baseline.
    """
    if baseline == "ewc":
        from models import EWCModel  # Replace with your implementation
    elif baseline == "si":
        from models import SIModel  # Replace with your implementation
    
    results = {}

    # Initialize model
    model = EWCModel(input_size=28*28, hidden_size=256, output_size=10) if baseline == "ewc" else SIModel(input_size=28*28, hidden_size=256, output_size=10)

    # Training loop
    for task_id, train_loader in enumerate(train_loaders):
        print(f"\nTraining on Task {task_id + 1} ({learning_scenario}-incremental, {baseline})")

        if baseline == "ewc":
            model.train_ewc(train_loader)  # Train with EWC
            model.update_fisher_information(train_loader)  # Update Fisher matrix
        elif baseline == "si":
            model.train_si(train_loader)  # Train with SI
            model.update_omega(train_loader)  # Update parameter importance
        else:
            model.train(train_loader)  # Vanilla SGD or other baselines

        # Evaluation
        task_results = {}
        for test_task_id, test_loader in enumerate(test_loaders):
            accuracy = model.evaluate(test_loader)  # Evaluate model
            task_results[f"Task {test_task_id + 1}"] = accuracy
            print(f"Accuracy on Task {test_task_id + 1}: {accuracy:.2f}%")

        results[f"Task {task_id + 1}"] = task_results

    return results


# Domain-incremental data loaders
train_loaders, test_loaders = get_data_separate_domain_incremental(batch_size=64)

# Train and evaluate EWC
domain_ewc_results = train_and_evaluate(
    model=None,
    train_loaders=train_loaders,
    test_loaders=test_loaders,
    learning_scenario="domain",
    baseline="ewc"
)


# Task-incremental data loaders
train_loaders, test_loaders = get_data_separate_dynamic(batch_size=64, num_tasks=2)

# Train and evaluate SI
task_si_results = train_and_evaluate(
    model=None,
    train_loaders=train_loaders,
    test_loaders=test_loaders,
    learning_scenario="task",
    baseline="si"
)


import json

# Save results for analysis
all_results = {
    "domain_incremental": {
        "ewc": domain_ewc_results,
        "si": domain_si_results
    },
    "task_incremental": {
        "ewc": task_ewc_results,
        "si": task_si_results
    }
}

with open("results.json", "w") as f:
    json.dump(all_results, f)
