## Graph QA Benchmark

The raw datasets can be downloaded from this [Hugging Face dataset](https://huggingface.co/datasets/baharef/GraphQA). It can be done directly, or by running:
```
python3 -m src.experiments.graphqa.download
```

Preparing the dataset for training GTLM can be performed by running the following python script:

```
python3 -m src.experiments.graphqa.process_dataset
```

This will create a `processed_datasets` folder inside of this directory.

You may now train the model on some specific task either by running this command:
```
python3 -m src.experiments.graphqa \
    --without={None/spd/rrwp/magnetic} \
    --graph_type={standard/indicence} \
    --task={chosen_task} \
    --lora_r=R

```
Available tasks to train the model on are: node_count, edge_count, cycle_check, triangle_counting, node_degree, connected_nodes, reachability, edge_existence, shortest_path.

The other option is to directly run the entire experiment with multi-seed analysis and an ablation study at once, by running:
```
python3 -m src.experiments.graphqa.full_experiment
```