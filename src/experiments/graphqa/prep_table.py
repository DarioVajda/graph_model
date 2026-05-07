import json
import math
import re
from collections import defaultdict
import os

# Define the relative path to the results
JSON_PATH = "./src/experiments/graphqa/results.json"

def compute_stats(values):
    """Computes the mean and sample standard deviation as percentages."""
    if len(values) < 3:
        return "?"
    
    mean = sum(values) / len(values)
    # Sample variance (n-1 degrees of freedom)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    std = math.sqrt(variance)
    
    return mean * 100, std * 100

def parse_baseline(b):
    """Parses static baseline string into a dictionary with value and suffix."""
    if isinstance(b, (int, float)):
        return {'val': float(b), 'suffix': ''}
    m = re.match(r"([0-9.]+)(.*)", str(b))
    return {'val': float(m.group(1)), 'suffix': m.group(2)}

def main():
    if not os.path.exists(JSON_PATH):
        print(f"Error: Could not find {JSON_PATH}")
        return 
        
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # Dictionary to hold arrays of accuracies
    parsed_data = defaultdict(list)
    regex = re.compile(r"^GraphQA_(base|no-magnetic|no-rrwp|no-spd)_(.*?)_(incidence|standard)(?:_v\d+)?$")
    
    for key, val in data.items():
        match = regex.match(key)
        if match:
            config, task, fmt = match.groups()
            parsed_data[(config, task, fmt)].append(val)

    # Compute stats
    stats = {}
    for key, values in parsed_data.items():
        stats[key] = compute_stats(values)

    # Baseline data arrays
    baselines_data = {
        'node_count': [21.7, "79.2\\llnote{MPNN}", "91.2\\llnote{MHA}", "99.6\\llnote{NS}"],
        'edge_count': [12.4, "26.4\\llnote{MHA}", "36.8\\llnote{MPNN}", "42.6\\llnote{ES}"],
        'cycle_check': [76.0, "96.2\\llnote{MHA}", "96.4\\llnote{GCN}", "96.4\\llnote{ES}"],
        'triangle_counting': [1.5, "23.4\\llnote{HGT}", "26.6\\llnote{MHA}", "34.8\\llnote{MPNN}"],
        'node_degree': [14.0, "26.6\\llnote{HGT}", "55.2\\llnote{MHA}", "96.2\\llnote{MPNN}"],
        'connected_nodes': [14.7, "24.4\\llnote{MHA}", "25.0\\llnote{MPNN}", "26.4\\llnote{GCN}"],
        'reachability': [84.9, "93.2\\llnote{MHA}", "94.2\\llnote{NS}", "94.4\\llnote{HGT}"],
        'edge_existence': [44.5, "68.0\\llnote{GCN}", "71.8\\llnote{HGT}", "73.8\\llnote{MHA}"],
        'shortest_path': [11.5, "60.4\\llnote{GCN}", "60.8\\llnote{MHA}", "63.8\\llnote{MPNN}"],
    }

    # Structure to hold Table 1 columns for dynamic ranking
    table1_cols = {}
    all_tasks = list(baselines_data.keys())
    
    for task in all_tasks:
        # Load the 4 baseline rows
        table1_cols[task] = [parse_baseline(b) for b in baselines_data[task]]
        
        # Row 4: GTLM Incidence
        res_inc = stats.get(('base', task, 'incidence'), "?")
        if res_inc == "?":
            table1_cols[task].append({'val': None, 'suffix': ''})
        else:
            table1_cols[task].append({'val': res_inc[0], 'suffix': f"\\std{{{res_inc[1]:.1f}}}"})
            
        # Row 5: GTLM Standard
        res_std = stats.get(('base', task, 'standard'), "?")
        if res_std == "?":
            table1_cols[task].append({'val': None, 'suffix': ''})
        else:
            table1_cols[task].append({'val': res_std[0], 'suffix': f"\\std{{{res_std[1]:.1f}}}"})

    # Apply Bold (1st) and Underline (2nd) formatting automatically per column for Table 1
    for task in all_tasks:
        cells = table1_cols[task]
        
        # Find unique, rounded values to prevent floating point comparison errors
        valid_vals = set(round(c['val'], 1) for c in cells if c['val'] is not None)
        sorted_vals = sorted(list(valid_vals), reverse=True)
        
        max_val = sorted_vals[0] if len(sorted_vals) > 0 else None
        second_max = sorted_vals[1] if len(sorted_vals) > 1 else None
        
        for c in cells:
            if c['val'] is None:
                c['fmt'] = "?"
                continue
            
            rnd_val = round(c['val'], 1)
            # Match formatting (100.0 -> 100 for visual cleanliness as per prompt)
            val_str = "100" if rnd_val == 100.0 else f"{rnd_val:.1f}"
            
            if max_val is not None and rnd_val == max_val:
                val_str = f"\\textbf{{{val_str}}}"
            elif second_max is not None and rnd_val == second_max:
                val_str = f"\\underline{{{val_str}}}"
                
            c['fmt'] = val_str + c['suffix']

    # Helpers to extract Table 1 rows
    def get_row_t1(row_idx, tasks):
        return " & ".join([table1_cols[t][row_idx]['fmt'] for t in tasks])

    t1a_tasks = ['node_count', 'edge_count', 'cycle_check', 'triangle_counting']
    t1b_tasks = ['node_degree', 'connected_nodes', 'reachability', 'edge_existence', 'shortest_path']

    # Table 2 Formatting (No bold/underline, uses \pm for std deviation)
    def format_ablation_cell(config, task, fmt="standard"):
        res = stats.get((config, task, fmt), "?")
        if res == "?":
            return "?"
        mean, std = res
        val_str = "100" if round(mean, 1) == 100.0 else f"{mean:.1f}"
        return f"{val_str} $\\pm$ {std:.1f}"
    
    ablation_tasks = ['node_count', 'edge_count', 'cycle_check', 'triangle_counting', 
                      'node_degree', 'connected_nodes', 'reachability', 'edge_existence', 'shortest_path']

    def get_row_t2(config):
        return " & ".join([format_ablation_cell(config, t) for t in ablation_tasks])

    table_1 = r"""\begin{table}[ht]
    \centering
    \footnotesize % Scales down the text cleanly
    \renewcommand{\arraystretch}{0.9} % Compresses vertical space between rows
    \setlength{\tabcolsep}{4pt} % Tightens horizontal space between columns
    \caption{Accuracy ($\%$) Comparison on \textbf{GraphQA} tasks. The baselines are a Zero-Shot LLM and the top-3 highest scoring graph encoding methods used by GraphToken. The highest score in each task is \textbf{bold} and the second highest is \underline{underlined}.}
    \label{tab:graphqa-results}
    
    % --- FIRST PART: GRAPH TASKS ---
    \textbf{(a) Graph Tasks} \\
    \vspace{0.2em}
    \begin{tabular}{l | c c c c}
        \toprule
        \textbf{Method} & \textbf{Node Count} & \textbf{Edge Count} & \textbf{Cycle Check} & \textbf{Triangle Counting} \\
        \midrule
        Zero-Shot LLM       & """ + get_row_t1(0, t1a_tasks) + r""" \\
        3rd GraphToken      & """ + get_row_t1(1, t1a_tasks) + r""" \\
        2nd GraphToken      & """ + get_row_t1(2, t1a_tasks) + r""" \\
        1st GraphToken      & """ + get_row_t1(3, t1a_tasks) + r""" \\
        \midrule
        \textbf{GTLM} {\scriptsize(Incidence)} & """ + get_row_t1(4, t1a_tasks) + r""" \\
        \textbf{GTLM} {\scriptsize(Standard)}  & """ + get_row_t1(5, t1a_tasks) + r""" \\
        \bottomrule
    \end{tabular}

    \vspace{1em} % Minimal gap between sub-tables

    % --- SECOND PART: NODE & EDGE TASKS ---
    \textbf{(b) Node and Edge Tasks} \\
    \vspace{0.2em}
    \begin{tabular}{l | c c | c c c}
        \toprule
        & \multicolumn{2}{c|}{\textbf{Node Tasks}} & \multicolumn{3}{c}{\textbf{Edge Tasks}} \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-6}
        \textbf{Method} & \textbf{Node Degree} & \textbf{Connected Nodes \;\;} & \textbf{Reachability} & \textbf{Edge Existence} & \textbf{Shortest Path} \\
        \midrule
        Zero-Shot LLM       & """ + get_row_t1(0, t1b_tasks) + r""" \\
        3rd GraphToken      & """ + get_row_t1(1, t1b_tasks) + r""" \\
        2nd GraphToken      & """ + get_row_t1(2, t1b_tasks) + r""" \\
        1st GraphToken      & """ + get_row_t1(3, t1b_tasks) + r""" \\
        \midrule
        \textbf{GTLM} {\scriptsize(Incidence)} & """ + get_row_t1(4, t1b_tasks) + r""" \\
        \textbf{GTLM} {\scriptsize(Standard)}  & """ + get_row_t1(5, t1b_tasks) + r""" \\
        \bottomrule
    \end{tabular}
\end{table}"""

    table_2 = r"""\begin{table}[ht]
\centering
\caption{\textbf{Ablation study} results, comparing the accuracies ($\%$) of standard GTLM to its variants without each attention bias type individually.}
\label{tab:ablation-study}
\resizebox{\textwidth}{!}{
\begin{tabular}{l | *{4}{c} | *{2}{c} | *{3}{c}}
\toprule
& \multicolumn{4}{c}{\textbf{Graph Tasks}} & \multicolumn{2}{c}{\textbf{Node Tasks}} & \multicolumn{3}{c}{\textbf{Edge Tasks}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(lr){8-10}
\textbf{Method} & \textbf{Node Count} & \textbf{Edge Count} & \textbf{Cycle Check} & \textbf{Tri. Count} & \textbf{Node Deg.} & \textbf{Conn. Nodes} & \textbf{Reachability} & \textbf{Edge Exist.} & \textbf{Shortest Path} \\

\midrule
Standard     & """ + get_row_t2('base') + r""" \\
\midrule
w/o SPD      & """ + get_row_t2('no-spd') + r""" \\
w/o RRWP     & """ + get_row_t2('no-rrwp') + r""" \\
w/o Magnetic & """ + get_row_t2('no-magnetic') + r""" \\

\bottomrule
\end{tabular}
}
\end{table}"""

    print("="*60)
    print("TABLE 1: GraphQA Results")
    print("="*60)
    print(table_1)
    print("\n" * 3)
    print("="*60)
    print("TABLE 2: Ablation Study")
    print("="*60)
    print(table_2)

if __name__ == "__main__":
    main()