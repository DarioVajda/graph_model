import os
from tqdm import tqdm

from ...utils import TextGraphDataset

def load_dataset(path, train=True, val=True, test=True):
    datasets = {}
    splits = [ split for split, load in zip(['train', 'val', 'test'], [train, val, test]) if load ]
    for split in splits:
        print('-'*28+f"Loading {split} dataset"+'-'*28)
        dataset = None
        # read the folders in the {path}/{split} directory
        ds_folders_sorted = sorted(os.listdir(os.path.join(path, split)), key=lambda x: int(x.split('-')[0]))
        for ds_folder in ds_folders_sorted:
            curr_ds_path = os.path.join(path, split, ds_folder)
            ds = TextGraphDataset.load(curr_ds_path)
            ds.assign_label(split)
            if dataset is None:
                dataset = ds
            else:
                dataset += ds
        datasets[split] = dataset
        print('-'*20+f"Loaded {len(dataset)} samples for {split} dataset"+'-'*20)
        print()
    return datasets

def calc_total_len(graph):
    total_len = 0
    for node in range(graph['num_nodes']):
        total_len += len(graph['input_ids'][node])
    return total_len

def calc_avg_total_len(ds):
    total_len = 0
    num_samples = min(1000, len(ds))  # Limit to first 1000 samples for efficiency
    for i in tqdm(range(num_samples)):
        total_len += calc_total_len(ds[i])
    return total_len / num_samples if num_samples > 0 else 0

if __name__ == '__main__':
    paths = [
        './src/experiments/benchmarks/processed_data/pubmed_hops2_neighbors30',
        './src/experiments/benchmarks/processed_data/cora_hops2_neighbors30',
        './src/experiments/benchmarks/processed_data/ogbn-arxiv_hops2_neighbors30',
        './src/experiments/benchmarks/processed_data/reddit_hops2_neighbors15',
    ]
    for path in paths:
        datasets = load_dataset(path, train=True, val=False, test=False)
        print(datasets['train'])

        avg_len = calc_avg_total_len(datasets['train'])
        print(f"Dataset: {os.path.basename(path)}, Average Total Length: {avg_len}")
        print('='*80)
        print('='*80)

    # Results:
    # Dataset: pubmed_hops2_neighbors30,        Average Total Length: 947.23
    # Dataset: cora_hops2_neighbors30,          Average Total Length: 529.341
    # Dataset: ogbn-arxiv_hops2_neighbors30,    Average Total Length: 806.641
    # Dataset: reddit_hops2_neighbors15,        Average Total Length: 2756.897