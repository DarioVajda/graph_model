from ...utils import TextGraphDataset

from tqdm import tqdm

def load_graphqa_datasets(dataset_dir, train_tasks, test_tasks, graph_type):
    """
    Loads and merges TextGraphDatasets across multiple problem types.
    
    Returns:
        Tuple: (train_dataset, test_dataset) where each is a merged TextGraphDataset containing all specified problem types.
    """
    datasets = {}
    tasks = {
        "train": train_tasks,
        "test": test_tasks
    }
    
    for split in ["train", "test"]:
        split_datasets = []
        
        for problem_type in tasks[split]:
            path = f"{dataset_dir}/{graph_type}/{problem_type}/{split}"
            loaded_ds = TextGraphDataset.load(path)

            # check if the dataset has the expected label, and if not, assign it based on the problem and graph types and resave it
            expected_label = f"{graph_type}/{problem_type}"
            current_label = loaded_ds[0]['ds_label']
            if expected_label != current_label:
                print(f"Dataset at {path} has unexpected label '{current_label}'. Reassigning to expected label '{expected_label}'...")
                loaded_ds.assign_label(expected_label)
                loaded_ds.save(path) # overwrite the existing dataset with the updated labels

            split_datasets.append(loaded_ds)
        
        if not split_datasets:
            raise ValueError(f"No datasets found for split '{split}' with the specified problem types.")

        combined_ds = split_datasets[0]
        for ds in tqdm(split_datasets[1:], desc=f"Combining {split} datasets"):
            combined_ds = combined_ds + ds 
            
        datasets[split] = combined_ds
        
    return datasets["train"], datasets["test"]
    


if __name__ == "__main__":
    dataset_dir = "./src/experiments/graphqa/processed_datasets"
    train_problem_types = [ "connected_nodes", "disconnected_nodes", "cycle_check", "edge_count", "edge_existence", "node_classification", "node_count", "node_degree", "reachability", "shortest_path", "triangle_counting" ]
    test_problem_types = [ "connected_nodes", "disconnected_nodes", "cycle_check", "edge_count", "edge_existence", "node_classification", "node_count", "node_degree", "reachability", "shortest_path", "triangle_counting" ]
    standard_train_dataset, standard_test_dataset = load_graphqa_datasets(dataset_dir, train_problem_types, test_problem_types, "standard")
    incidence_train_dataset, incidence_test_dataset = load_graphqa_datasets(dataset_dir, train_problem_types, test_problem_types, "incidence")

    # compute the average total number of tokens in both the standard and incidence test datasets
    def compute_avg_tokens(dataset):
        total_tokens = 0
        for item in dataset:
            for node in range(len(item['input_ids'])):
                total_tokens += len(item['input_ids'][node])
        return total_tokens / len(dataset)

    # avg_tokens_standard = compute_avg_tokens(standard_test_dataset)
    # avg_tokens_incidence = compute_avg_tokens(incidence_test_dataset)
    # print(f"Average total number of tokens in standard test dataset: {avg_tokens_standard}")    # ~35  tokens
    # print(f"Average total number of tokens in incidence test dataset: {avg_tokens_incidence}")  # ~150 tokens
    