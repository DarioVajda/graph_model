"""
Multi-task loading of built GraphQA datasets.

`data.py` is what this experiment's own runs use (one task per run, three splits).
This module survives for the *merged multi-task* case — several tasks concatenated
into one dataset — which `graphqa_mag_khop` depends on.

For reference, average total tokens per example: ~35 (standard) / ~150 (incidence).
"""

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