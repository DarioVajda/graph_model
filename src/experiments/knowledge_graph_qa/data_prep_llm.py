from .data_gen import generate_dataset

import os
import random
from tqdm import tqdm
import json

# Disable tokenizer parallelism to make it safe for Python multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def serialize_graph(graph):
    # 1. Serialize the graph topology into a list of triplets
    triplets = []
    for u, v, data in graph.edges(data=True):
        relation = data['relation'].lower()
        # Create a natural language triplet (e.g., "(Ava Evans, works on, Strategic Gate)")
        triplets.append(f"({u}, {relation}, {v})")
    
    # Join all triplets into a single block of context text
    context_text = "Context Facts:\n" + "\n".join(triplets) + "\n\n"

    # 2. Append each question to the context
    serialized_prompts = []
    for func_name, (question, pointers, answer) in graph.graph['questions'].items():
        # Combine the context, the question, and the answer indicator
        full_prompt = f"{context_text}Q: {question}\nA: {answer}"
        serialized_prompts.append(full_prompt)

    return serialized_prompts

def serialize_graphs(graphs):
    textual_data = [ ]
    for graph in tqdm(graphs, desc="Serializing Graphs"):
        textual_data.extend(serialize_graph(graph))

    return textual_data


def tokenize_textual_data(textual_data, tokenizer):
    tokenized_data = []
    for text in tqdm(textual_data, desc="Tokenizing"):
        # tokens should be python lists
        tokens = tokenizer(
            text, 
            padding=False, 
            truncation=True, 
            max_length=32384,
            add_special_tokens=False,
        )['input_ids']
        tokenized_data.append(tokens)
    return tokenized_data

#region Extracting labels
import multiprocessing
from functools import partial

def _process_single_sequence(seq, target_seq):
    """
    Worker function to process a single list of integers.
    Searches backwards to find the target sequence quickly.
    """
    n = len(seq)
    m = len(target_seq)
    
    if m == 0 or n < m:
        raise ValueError("Target sequence is invalid or longer than the input sequence.")

    # 1. Search backwards (finding the LAST occurrence of the target)
    for i in range(n - m, -1, -1):
        # 2. Short-circuit check: only slice if the first token matches
        if seq[i] == target_seq[0] and seq[i:i+m] == target_seq:
            end_idx = i + m
            
            # 3. Fast list concatenation
            # Create a list of -100s and append the remainder of the original sequence
            return [-100] * end_idx + seq[end_idx:]
            
    for token_id in seq:
        print(f"{token_id}: {tokenizer.decode([token_id]).replace(' ', '_')}")
    raise ValueError(f"Target sequence {target_seq} not found in the example.")

def generate_labels(tokenized_sequences, target_seq=[32, 25], num_proc=None):
    """
    Parallelizes label generation with a real-time progress bar.
    """
    if num_proc is None:
        num_proc = max(1, multiprocessing.cpu_count() - 1)
        
    print(f"Generating labels across {num_proc} CPU cores...")
        
    worker_fn = partial(_process_single_sequence, target_seq=target_seq)
    
    # Calculate chunksize
    total_items = len(tokenized_sequences)
    chunksize = max(1, total_items // (num_proc * 4))
    
    # Use Process Pool
    with multiprocessing.Pool(processes=num_proc) as pool:
        labels = list(tqdm(
            pool.imap(worker_fn, tokenized_sequences, chunksize=chunksize),
            total=total_items,
            desc="Masking Labels"
        ))
        
    return labels
#endregion

if __name__ == "__main__":
    VAL_COUNT = 150
    TEST_COUNT = 250
    TRAIN_COUNT = 1200

    MIN_NODES = 50
    MAX_NODES = 100
    raw_train_graphs, raw_val_graphs, raw_test_graphs = generate_dataset(train_count=TRAIN_COUNT, val_count=VAL_COUNT, test_count=TEST_COUNT, min_nodes=MIN_NODES, max_nodes=MAX_NODES)
    print(f"Generated {len(raw_train_graphs)} training graphs, {len(raw_val_graphs)} validation graphs, and {len(raw_test_graphs)} test graphs with node counts between {MIN_NODES} and {MAX_NODES}.")

    datasets = {
        'val': raw_val_graphs,
        # 'test': raw_test_graphs,
        # 'train': raw_train_graphs,
    }

    save_path = f"./src/experiments/knowledge_graph_qa/text_datasets/dataset_{MIN_NODES}-{MAX_NODES}"
    
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    print("Tokenizer loaded.")

    for split, graphs in datasets.items():
        print(f"\n{split.upper()} DATASET:")
        textual_data = serialize_graphs(graphs)

        tokenized_data = tokenize_textual_data(textual_data, tokenizer)
        labels = generate_labels(tokenized_data, target_seq=[32, 25], num_proc=16) # this represents "A:"

        dataset = [
            {
                'text': text,
                'input_ids': tokenized,
                'labels': label,
            } for text, tokenized, label in zip(textual_data, tokenized_data, labels)
        ]

        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"{split}_dataset.jsonl")
        with open(save_file, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")

        # compute the average tokenized length on the validation dataset, as it is the smallest
        if split == 'val':
            total_tokens = 0
            total_examples = 0
            for i in range(0, len(dataset), 5):
                example = dataset[i]
                total_tokens += len(example['input_ids'])
                total_examples += 1
            avg_tokens = total_tokens / total_examples if total_examples > 0 else 0
            print(f"Average tokenized length in {split} dataset: {avg_tokens:.2f} tokens")
            # OUTPUT: Average tokenized length in val dataset: 3290.71 tokens