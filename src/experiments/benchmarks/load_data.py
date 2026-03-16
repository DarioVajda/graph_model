import os

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


if __name__ == '__main__':
    path = './src/experiments/benchmarks/processed_data/ogbn-arxiv_hops2_neighbors30'
    datasets = load_dataset(path, train=True, val=True, test=True)
    print(datasets['train'])
    print(datasets['val'])
    print(datasets['test'])