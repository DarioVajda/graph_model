import json, os

def load_graph_dataset_split(path, split):
    # lazy import to avoid excessive waiting time when not needed
    from ...utils.text_graph_dataset import TextGraphDataset

    # list all files in the path that contain the substring {split}
    files = [f for f in os.listdir(path) if split in f and f.endswith('.gtds')]
    if len(files) > 1:
        files = sorted(files, key=lambda x: int(x.split('-')[-1].split('.')[0])) # sort by the number after the last '-' and before '.gtds'

    if not files:
        raise ValueError(f"No dataset files found for split '{split}' in path '{path}'")

    # load all files and add them up together
    dataset = None
    for file in files:
        curr_dataset = TextGraphDataset.load(os.path.join(path, file))
        if dataset is None:
            dataset = curr_dataset
        else:
            dataset = dataset + curr_dataset
        
    print(f"Loaded dataset for split '{split}' from {len(files)} files with a total of {len(dataset)} examples.")
    return dataset

def load_text_dataset_split(path, split):
    file = os.path.join(path, f"{split}_dataset.jsonl")
    if not os.path.exists(file):
        raise ValueError(f"No dataset file found for split '{split}' at path '{file}'")

    dataset = []
    with open(file, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    
    print(f"Loaded text dataset for split '{split}' from file '{file}' with a total of {len(dataset)} examples.")
    return dataset

def load_dataset(path, type='graph'):
    if type == 'graph':
        return load_graph_dataset_split(path, split='train'), load_graph_dataset_split(path, split='val'), load_graph_dataset_split(path, split='test')
    elif type == 'text':
        return load_text_dataset_split(path, split='train'), load_text_dataset_split(path, split='val'), load_text_dataset_split(path, split='test')
    else:
        raise ValueError(f"Invalid dataset type '{type}'. Expected 'graph' or 'text'.")

if __name__ == "__main__":
    ds_type = 'graph'
    # dataset_path = f"./src/experiments/knowledge_graph_qa/{ds_type}_datasets/dataset_50-100"
    dataset_path = f"./src/experiments/knowledge_graph_qa/family_tree_graph_dataset"
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_path, type=ds_type)

    print(f"{ds_type.upper()} DATASET SIZES:")
    print(f"Training dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")
    print(f"Test dataset: {len(test_dataset)} examples")

    datasets = {
        'val': val_dataset,
        'test': test_dataset,
        'train': train_dataset,
    }

    from tqdm import tqdm
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # for split, ds in datasets.items():
    #     print(f"--- Checking dataset split: {split} ---")
    #     # check if each example has exactly 1 unmasked token in the labels (example['labels'] is a tensor with -100 for masked tokens and a single non-negative integer for the unmasked token)
    #     for i, example in tqdm(enumerate(ds), desc=f"Checking {split} dataset", total=len(ds)):
    #         if not (example['labels'][-1] != -100 and example['labels'][-2] == -100):
    #             print(f"Example {i} in split '{split}' has {example['labels'].shape[0] - (example['labels'] == -100).sum()} unmasked tokens instead of 1.")
    #             print(f"Labels: {example['labels']}")
    #             for id in example['labels']:
    #                 print(f"{id}: {tokenizer.decode(id) if id != -100 else '[MASKED]'}")
    #             print('='*70)
    #     print('-----------------------------------------')  
    
    import torch
    train_ds = datasets['train']

    # check average length of input_ids in the training dataset
    total_length = 0
    for example in tqdm(train_ds, desc="Checking average length of input_ids"):
        for node_input_ids in example['input_ids']:
            total_length += len(node_input_ids)
    avg_length = total_length / len(train_ds)
    print(f"Average length of input_ids in the training dataset: {avg_length}")

    # print(train_ds[0].keys())
    # for key in ['input_ids', 'labels', 'shortest_path_dists', 'rrwp', 'magnetic_V', 'magnetic_lambdas']:
    #     print(f"{key}: {type(train_ds[0][key])}, example value: {train_ds[0][key]}")
    # exit()

    # check if there are any NaN values in any of the tensors in the first 200 examples of the training dataset
    # fields of interest: 'input_ids', 'labels', 'shortest_path_dists', 'rrwp', 'magnetic_V', and 'magnetic_lambdas'
    # for i, example in tqdm(enumerate(train_ds), desc="Checking for NaN values in training dataset", total=len(train_ds)):
    #     for field in ['labels', 'shortest_path_dists', 'rrwp', 'magnetic_V', 'magnetic_lambdas']:
    #         if field in example:
    #             tensor = example[field]
    #             if isinstance(tensor, list):
    #                 tensor = torch.tensor(tensor)
    #             if torch.isnan(tensor).any():
    #                 print(f"Example {i} has NaN values in field '{field}'.")
    #                 print(f"Tensor: {tensor}")
    #                 print('='*70)