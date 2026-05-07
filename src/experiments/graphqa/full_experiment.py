from .train import run_training

def load_existing_results():
    # read the json file at path ./src/experiments/graphqa/results.json
    import json
    import os
    results_path = os.path.join(os.path.dirname(__file__), 'results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:        
        results = {}
    return results

if __name__ == "__main__":
    without_options = [
        None,
        "spd",
        "rrwp",
        "magnetic",
    ]
    graph_type_options = [
        "standard",
        "incidence",
    ]
    task_options = [
        "node_count",
        "edge_count",
        "cycle_check",
        "triangle_counting",
        "node_degree",
        "connected_nodes",
        "reachability",
        "edge_existence",
        "shortest_path",
    ]
    lora_r = 16
    def get_run_name(without, graph_type, task):
        without_str = "no-" + without if without is not None else "base"
        return f"GraphQA_{without_str}_{task}_{graph_type}"
    
    existing_results = load_existing_results().items()
    run_index = 0
    
    for graph_type in graph_type_options:
        for task in task_options:
            for without in without_options:
                for run in range(3):  # Run each experiment 3 times
                    if len(existing_results) > run_index:
                        print(f"Skipping {run_index+1}. run since we already have {len(existing_results)} existing results.")
                        run_index += 1
                        continue

                    if without is not None and graph_type == "incidence":
                        print(f"Skipping ablation study for incidence graphs (we got without={without} and graph_type={graph_type})")
                        continue
                    run_name = get_run_name(without, graph_type, task)
                    print("." * 100)
                    print(f"Running experiment {run + 1}: {run_name}")
                    print("." * 100)
                    run_training(
                        without=without,
                        graph_type=graph_type,
                        task=task,
                        lora_r=lora_r,
                        run_name=run_name,
                        seed=42 + run,
                        use_wandb=True,
                    )

                    run_index += 1
                    print(f"Completed {run_index} runs so far.")