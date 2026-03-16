"""
This is a file which I had to write later, as I forgot to add the "add_eos=True" flag in the initial data processing script.
"""
import os

from transformers import AutoTokenizer

from ...utils import TextGraphDataset
from .process_data import GetGraphLabels

if __name__ == '__main__':
    root_path = './src/experiments/benchmarks/processed_data'
    datasets = os.listdir(root_path)
    splits = ['train', 'val', 'test']

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    get_graph_labels = GetGraphLabels(question_end=[ 32, 25 ])

    for dataset in datasets:
        print('-'*28+f"Loading {dataset} dataset"+'-'*28)
        for split in splits:
            print('-'*20+f"Loading {split} split"+'-'*20)
            dataset_path = os.path.join(root_path, dataset, split)
            ds_folders_sorted = sorted(os.listdir(dataset_path), key=lambda x: int(x.split('-')[0]))
            for ds_folder in ds_folders_sorted:
                curr_ds_path = os.path.join(dataset_path, ds_folder)
                ds = TextGraphDataset.load(curr_ds_path)
                ds.tokenize(tokenizer, max_length=32_768, add_eos=True)
                ds.compute_labels(get_graph_labels)
                ds.save(curr_ds_path)
                print(f"Fixed {curr_ds_path}")
                print()
            print('-'*50)
        print('='*10)