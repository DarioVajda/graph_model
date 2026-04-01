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


if __name__ == '__main__':
    def calc_total_len(graph):
        total_len = 0
        for node in range(graph['num_nodes']):
            total_len += len(graph['input_ids'][node])
        return total_len, graph['num_nodes']

    def calc_avg_total_len(ds):
        total_len = 0
        node_count = 0
        num_samples = min(1000, len(ds))  # Limit to first 1000 samples for efficiency
        for i in range(num_samples):
            tl, nc = calc_total_len(ds[i])
            total_len += tl
            node_count += nc
        return total_len / num_samples if num_samples > 0 else 0, total_len / node_count if node_count > 0 else 0


    paths = [
        # CORA
        './src/experiments/benchmarks/processed_data/cora_hops2_neighbors30_target_abstract',
        './src/experiments/benchmarks/processed_data/cora_hops2_neighbors60_target_abstract',
        './src/experiments/benchmarks/processed_data/cora_hops2_neighbors111_target_abstract',
        # './src/experiments/benchmarks/processed_data/ogbn-arxiv_hops2_neighbors30',
        # './src/experiments/benchmarks/processed_data/reddit_hops2_neighbors15',
        # './src/experiments/benchmarks/processed_data/cora_hops2_neighbors15_neighbor_abstracts',
        # './src/experiments/benchmarks/processed_data/cora_hops2_neighbors30_random_abstracts_p0.5',
        # './src/experiments/benchmarks/processed_data/cora_hops2_neighbors30_random_abstracts_p0.3',
        # './src/experiments/benchmarks/processed_data/cora_hops2_neighbors30_random_abstracts_p0.2',

        # PUBMED
        './src/experiments/benchmarks/processed_data/pubmed_hops2_neighbors30',
        './src/experiments/benchmarks/processed_data/pubmed_hops2_neighbors60_target_abstract',

        # REDDIT
        './src/experiments/benchmarks/processed_data/reddit_hops2_neighbors15_full_text',
        # './src/experiments/benchmarks/processed_data/reddit_hops2_neighbors15_truncated_text_128',
        './src/experiments/benchmarks/processed_data/reddit_hops2_neighbors30_truncated_text_128',
        # './src/experiments/benchmarks/processed_data/reddit_hops2_neighbors45_truncated_text_128',
        './src/experiments/benchmarks/processed_data/reddit_hops2_neighbors60_truncated_text_64',

        # OGBN-ARXIV
        './src/experiments/benchmarks/processed_data/ogbn-arxiv_hops2_neighbors30',
        './src/experiments/benchmarks/processed_data/ogbn-arxiv_hops2_neighbors60_target_abstract',
        './src/experiments/benchmarks/processed_data/ogbn-arxiv_hops2_neighbors20_random_abstracts_p0.2'

    ]
    results = {}
    for path in paths:
        datasets = load_dataset(path, train=True, val=False, test=False)
        # print(datasets['train'])

        avg_per_graph_len, avg_per_node_len = calc_avg_total_len(datasets['train'])
        print(f"Dataset: {os.path.basename(path)}\nAverage Total Length: {avg_per_graph_len}\nAverage Length per Node: {avg_per_node_len}")
        print('='*80)
        results[os.path.basename(path)] = (len(datasets['train']), datasets['train'].per_graph_versions, avg_per_graph_len, avg_per_node_len)
        # print('='*80)

    print()
    print('-'*61+"Summary of Results"+'-'*61)
    print('-'*140)
    print("|                      Dataset                       | Dataset Size     | Graph Versions     | Avg Total Length     | Avg Length per Node  |")
    for dataset_name, (ds_size, per_graph_versions, avg_graph_len, avg_node_len) in results.items():
        print(f"| {dataset_name:<50} | {ds_size:<16} | {per_graph_versions:<18} | {avg_graph_len:<20.2f} | {avg_node_len:<20.2f} |")
    print('-'*140)