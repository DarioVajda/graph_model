import networkx as nx
import random
import json
import numpy as np
from tqdm import tqdm

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def generate_names():
    first_names = [
        'Joe', 'Mike', 'Adam', 'Eve', 'Elisabeth', 'Ana', 'Thomas', 'Olivia', 
        'Emma', 'Amelia', 'Sophia', 'James', 'Robert', 'John', 'David', 
        'William', 'Noah', 'Lucas', 'Mia', 'Alexander', 'Charlotte', 'Benjamin', 
        'Harper', 'Elijah', 'Evelyn', 'Liam', 'Isabella', 'Oliver', 'Ava', 
        'Mason', 'Emily', 'Logan', 'Abigail', 'Ethan', 'Ella', 'Jacob', 
        'Madison', 'Michael', 'Elizabeth', 'Daniel', 'Sofia', 'Henry', 'Avery', 
        'Scarlett', 'Sebastian', 'Victoria', 'Aiden', 'Aria', 'Matthew', 'Grace', 
        'Samuel', 'Chloe', 'Camila', 'Joseph', 'Penelope', 'Carter', 'Riley', 
        'Owen', 'Layla'
    ]
    last_names = [
        'Shmoe', 'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 
        'Miller', 'Davis', 'Rodrigues', 'Martinez', 'Taylor', 'Wilson', 'Moore', 
        'Jackson', 'Lee', 'White', 'Anderson', 'Thomas', 'Hernandez', 'Martin', 
        'Thompson', 'Lopez', 'Gonzalez', 'Harris', 'Clark', 'Lewis', 'Robinson', 
        'Walker', 'Perez', 'Hall', 'Young', 'Allen', 'Sanchez', 'Wright', 'King', 
        'Scott', 'Green', 'Baker', 'Adams', 'Nelson', 'Hill', 'Ramirez', 'Campbell', 
        'Mitchell', 'Roberts', 'Carter', 'Phillips', 'Evans', 'Turner', 'Torres', 
        'Parker', 'Collins', 'Edwards', 'Stewart', 'Flores', 'Morris', 'Nguyen', 
        'Murphy', 'Rivera'
    ]
    # 60 * 60 = 3,600 unique full names
    return [f"{f} {l}" for f in first_names for l in last_names]

def generate_projects():
    prefixes = [ "Project", "Operation", "Initiative", "Taskforce", "Plan", "Alpha", "Delta", "Omega", "Blue", "Silver", "Crimson", "Obsidian", "Quantum", "Midnight", "Aero", "Solar", "Deep", "Global", "Strategic", "Vanguard", "Titan", "Neon" ]
    suffixes = [ "Horizon", "Shield", "Pulse", "Bridge", "Catalyst", "Forge", "Nexus", "Sentinel", "Zenith", "Prism", "Sentry", "Vault", "Flow", "Net", "Sky", "Anchor", "Beacon", "Core", "Drift","Engine", "Gate", "Haven", "Impact", "Link", "Mind" ]
    return [f"{p} {s}" for p in prefixes for s in suffixes]

def generate_resources():
    adjectives = [ 'super', 'quantum', 'new', 'neural', 'parallel', 'cryogenic', 'synthetic', 'autonomous', 'optical', 'distributed' ]
    nouns = [ 'computer', 'accelerator', 'machine', 'processor', 'engine', 'cluster', 'network', 'interface', 'terminal', 'mainframe' ]
    return [f"{a} {n}" for a in adjectives for n in nouns]

#region Questions
# --------------------------- One Hop Reasoning Questions ---------------------------
def works_on(graph, person, project):
    question = f"Does {person} work on {project}?"
    if graph.nodes[person]['type'] == 'person' and graph.nodes[project]['type'] == 'project' and graph.has_edge(person, project) and graph.edges[person, project]['relation'] == 'WORKS_ON':
        return question, 'Yes'
    else:
        return question, 'No'

def reports_to(graph, person1, person2):
    question = f"Does {person1} report to {person2}?"
    if graph.nodes[person1]['type'] == 'person' and graph.nodes[person2]['type'] == 'person' and graph.has_edge(person1, person2) and graph.edges[person1, person2]['relation'] == 'REPORTS_TO':
        return question, 'Yes'
    else:
        return question, 'No'

def requires(graph, project, resource):
    question = f"Does {project} require {resource}?"
    if graph.nodes[project]['type'] == 'project' and graph.nodes[resource]['type'] == 'resource' and graph.has_edge(project, resource) and graph.edges[project, resource]['relation'] == 'REQUIRES':
        return question, 'Yes'
    else:
        return question, 'No'

def can_access(graph, person, resource):
    question = f"Can {resource} be accessed by {person}?"
    if graph.nodes[resource]['type'] == 'resource' and graph.nodes[person]['type'] == 'person' and graph.has_edge(person, resource) and graph.edges[person, resource]['relation'] == 'CAN_ACCESS':
        return question, 'Yes'
    else:
        return question, 'No'
# -----------------------------------------------------------------------------------

# --------------------------- Two Hop Reasoning Questions ---------------------------
def works_on_project_which_requires(graph, person, resource):
    quesiton = f"Does {person} work on a project which requires {resource}?"
    if graph.nodes[person]['type'] != 'person' or graph.nodes[resource]['type'] != 'resource':
        return quesiton, 'No'
    
    for neighbor in graph.neighbors(person):
        if graph.edges[person, neighbor]['relation'] == 'WORKS_ON' and graph.nodes[neighbor]['type'] == 'project':
            project = neighbor
            if graph.has_edge(project, resource) and graph.edges[project, resource]['relation'] == 'REQUIRES':
                return quesiton, 'Yes'
    return quesiton, 'No'

def reports_to_who_works_on(graph, person, project):
    question = f"Does {person} report to someone who works on {project}?"
    if graph.nodes[person]['type'] != 'person' or graph.nodes[project]['type'] != 'project':
        return question, 'No'
    
    for neighbor in graph.neighbors(person):
        if graph.edges[person, neighbor]['relation'] == 'REPORTS_TO' and graph.nodes[neighbor]['type'] == 'person':
            boss = neighbor
            if graph.has_edge(boss, project) and graph.edges[boss, project]['relation'] == 'WORKS_ON':
                return question, 'Yes'
    return question, 'No'
    
def report_to_same_person(graph, person1, person2):
    question = f"Do {person1} and {person2} report to the same person?"
    if graph.nodes[person1]['type'] != 'person' or graph.nodes[person2]['type'] != 'person':
        return question, 'No'
    
    boss1 = None
    boss2 = None
    
    for neighbor in graph.neighbors(person1):
        if graph.edges[person1, neighbor]['relation'] == 'REPORTS_TO' and graph.nodes[neighbor]['type'] == 'person':
            boss1 = neighbor
            break
            
    for neighbor in graph.neighbors(person2):
        if graph.edges[person2, neighbor]['relation'] == 'REPORTS_TO' and graph.nodes[neighbor]['type'] == 'person':
            boss2 = neighbor
            break
            
    if boss1 is not None and boss1 == boss2:
        return question, 'Yes'
    else:
        return question, 'No'

def projects_require_same_resource(graph, project1, project2):
    question = f"Do {project1} and {project2} require a same resource?"
    if graph.nodes[project1]['type'] != 'project' or graph.nodes[project2]['type'] != 'project':
        return question, 'No'
    
    resources1 = set()
    resources2 = set()
    
    for neighbor in graph.neighbors(project1):
        if graph.edges[project1, neighbor]['relation'] == 'REQUIRES' and graph.nodes[neighbor]['type'] == 'resource':
            resources1.add(neighbor)
            
    for neighbor in graph.neighbors(project2):
        if graph.edges[project2, neighbor]['relation'] == 'REQUIRES' and graph.nodes[neighbor]['type'] == 'resource':
            resources2.add(neighbor)
    
    if len(resources1.intersection(resources2)) > 0:
        return question, 'Yes'
    else:
        return question, 'No'
# -----------------------------------------------------------------------------------

# -------------------------- Multi-Hop Reasoning Questions --------------------------
def is_person_ceo(graph, person):
    question = f"Is {person} the CEO (i.e., has no boss)?"
    if graph.nodes[person]['type'] != 'person':
        return question, 'No'
    
    for neighbor in graph.neighbors(person):
        if graph.edges[person, neighbor]['relation'] == 'REPORTS_TO' and graph.nodes[neighbor]['type'] == 'person':
            return question, 'No'
    return question, 'Yes'

def are_project_requirements_accessible(graph, project):
    question = f"Are all resources required by {project} accessible by the people working on it?"
    
    # 1. Resources the project needs (Project -> Resource)
    required_resources = { n for n in graph.neighbors(project) if graph.edges[project, n]['relation'] == 'REQUIRES' }
    
    if not required_resources:
        return question, 'Yes'

    # 2. People working on the project (Person -> Project, so use predecessors)
    workers = [ n for n in graph.predecessors(project) if graph.edges[n, project]['relation'] == 'WORKS_ON' ]
    
    # 3. Check if the set of resources these workers can access covers the requirements
    accessible_by_team = set()
    for person in workers:
        # Get all resources this person has access to (Person -> Resource)
        for target in graph.neighbors(person):
            if graph.edges[person, target]['relation'] == 'CAN_ACCESS':
                accessible_by_team.add(target)

    if required_resources.issubset(accessible_by_team):
        return question, 'Yes'
    return question, 'No'

def does_some_subordinate_work_on_project(graph, person, project):
    question = f"Does {person} or any of their subordinates work on {project}?"
    
    # DFS to go DOWN the chain. 
    # Since edges are (subordinate -> boss), we must look at PREDECESSORS to go down.
    visited = set()
    stack = [person]

    while stack:
        current = stack.pop()
        if current in visited: continue
        visited.add(current)

        # Check if current person works on the project (Out-edge to project)
        if graph.has_edge(current, project) and graph.edges[current, project]['relation'] == 'WORKS_ON':
            return question, 'Yes'

        # Add subordinates (those who report to 'current')
        for sub in graph.predecessors(current):
            if graph.edges[sub, current]['relation'] == 'REPORTS_TO':
                stack.append(sub)

    return question, 'No'
# -----------------------------------------------------------------------------------

def random_person(people, projs, res):
    return random.choice(people)
def random_project(people, projs, res):
    return random.choice(projs)
def random_resource(people, projs, res):    
    return random.choice(res)

#endregion

class KnowledgeGraphGenerator:
    def __init__(self):
        self.names = generate_names()
        self.projects = generate_projects()
        self.resources = generate_resources()


    def generate_questions(self, graph, people, projs, res, train=True):
        questions = {} # map of func_name -> (question, answer) pairs
        problems = [
            # Easy
            # (works_on, [random_person, random_project]),
            # (reports_to, [random_person, random_person]),
            # (requires, [random_project, random_resource]),
            # (can_access, [random_person, random_resource]),
            # Medium
            # (works_on_project_which_requires, [random_person, random_resource]),
            (reports_to_who_works_on, [random_person, random_project]),
            (report_to_same_person, [random_person, random_person]),
            (projects_require_same_resource, [random_project, random_project]),
            # Hard
            (is_person_ceo, [random_person]),
            (are_project_requirements_accessible, [random_project]),
            (does_some_subordinate_work_on_project, [random_person, random_project])
        ]

        # Choose a random label with probability 1/2 for yes and 1/2 for no for each problem type
        # Try to get the desired label maximum of 5 times (to balance the training dataset label counts) for training and 1 time for testing (to have the natural distribution of answers))
        for problem, arg_funcs in problems:
            label = 'Yes' if random.random() < 0.5 else 'No'
            attempts = 500 if train else 1
            success = False
            attempts_left = attempts
            while attempts_left > 0:
                args = [func(people, projs, res) for func in arg_funcs]
                # check if there are duplicate arguments (e.g. same person chosen twice for reports_to) and if so, skip this iteration and try again to avoid trivial questions
                if len(set(args)) < len(args):
                    continue
                
                attempts_left -= 1
                question, answer = problem(graph, *args)
                if answer == label:
                    questions[problem.__name__] = (question, args, answer)
                    success = True
                    break
            if not success:
                questions[problem.__name__] = (question, args, answer)
        
        return questions

    def generate(self, example_count=1, min_nodes=20, max_nodes=100, train=True):
        graphs = []
        print(f"Generating {example_count} graphs with node counts between {min_nodes} and {max_nodes}...")
        for _ in tqdm(range(example_count), desc="Generating Graphs"):
            G = nx.DiGraph()
            num_nodes = random.randint(min_nodes, max_nodes)
            
            # 1. Distribute node types
            p_count = int(num_nodes * 0.55)
            proj_count = int(num_nodes * 0.2)
            res_count = num_nodes - p_count - proj_count

            people = random.sample(self.names, p_count)
            projs = random.sample(self.projects, proj_count)
            res = random.sample(self.resources, res_count)

            # 2. Add nodes with attributes
            for p in people: G.add_node(p, type='person')
            for pr in projs: G.add_node(pr, type='project')
            for r in res: G.add_node(r, type='resource')

            # 3. WORKS_ON (Geometric p=2/3)
            for p in people:
                num_projs = min(len(projs), np.random.geometric(p=2/3))
                for pr in random.sample(projs, num_projs):
                    G.add_edge(p, pr, relation='WORKS_ON')

            # 4. REPORTS_TO (Hierarchy)
            for i, p in enumerate(people):
                # if i > 0 and random.random() > 0.3:
                if i > 0:
                    boss = random.choice(people[:i]) # Only report to someone "above" you
                    G.add_edge(p, boss, relation='REPORTS_TO')
            
            # 5. REQUIRES (Geometric p=3/5)
            for pr in projs:
                num_res = min(len(res), np.random.geometric(p=3/5))
                for r in random.sample(res, num_res):
                    G.add_edge(pr, r, relation='REQUIRES')
            
            # 6. CAN_ACCESS (Geometric p=3/5)
            for p in people:
                num_res = min(len(res), np.random.geometric(p=3/5))
                for r in random.sample(res, num_res):
                    G.add_edge(p, r, relation='CAN_ACCESS')
            
            questions = self.generate_questions(G, people, projs, res, train)
            G.graph['questions'] = questions

            graphs.append(G)
        return graphs

def print_example(graph):
    print("Nodes:")
    for node, data in graph.nodes(data=True):
        print(f"  {data['type']}: {node}")
    print("\nEdges:")
    for u, v, data in graph.edges(data=True):
        print(f"  {u} --{data['relation']}--> {v}")
    print("\nSample Questions:")
    for func_name, (q, pointers, a) in graph.graph['questions'].items():
        print(f"{func_name}:\nQ: {q}\nPointers: {pointers}\nA: {a}\n")

def compute_label_stats(graphs):
    label_counts = {} # problem_type -> {'Yes': count, 'No': count}
    total_counts = {'Yes': 0, 'No': 0}
    for graph in graphs:
        for func_name, (q, pointers, a) in graph.graph['questions'].items():
            if func_name not in label_counts:
                label_counts[func_name] = {'Yes': 0, 'No': 0}
            label_counts[func_name][a] += 1
            total_counts[a] += 1
    
    label_counts['OVERALL'] = total_counts
    return label_counts

def print_label_stats(label_stats):
    print("Label Distribution:")
    for func_name, counts in label_stats.items():
        total = counts['Yes'] + counts['No']
        yes_pct = counts['Yes'] / total * 100 if total > 0 else 0
        no_pct = counts['No'] / total * 100 if total > 0 else 0
        print(f"{func_name:<40}: Yes={counts['Yes']} ({yes_pct:.1f}%), No={counts['No']} ({no_pct:.1f}%)")

def print_size_stats(graphs):
    # print distribution of node counts, edge counts and total node+edge counts
    node_counts = [graph.number_of_nodes() for graph in graphs]
    edge_counts = [graph.number_of_edges() for graph in graphs]
    total_counts = [node_counts[i] + edge_counts[i] for i in range(len(graphs))]

    print("=" * 50)
    print("Graph Size Statistics:")
    print("=" * 50)
    print("Node Count: mean={:.1f}, std={:.1f}, min={}, max={}".format(np.mean(node_counts), np.std(node_counts), np.min(node_counts), np.max(node_counts)))
    print("Edge Count: mean={:.1f}, std={:.1f}, min={}, max={}".format(np.mean(edge_counts), np.std(edge_counts), np.min(edge_counts), np.max(edge_counts)))
    print("Total Count: mean={:.1f}, std={:.1f}, min={}, max={}".format(np.mean(total_counts), np.std(total_counts), np.min(total_counts), np.max(total_counts)))
    print("=" * 50)

def generate_dataset(train_count=1000, val_count=200, test_count=200, min_nodes=20, max_nodes=100):
    train_graphs = KnowledgeGraphGenerator().generate(train_count, min_nodes=min_nodes, max_nodes=max_nodes, train=True)
    val_graphs = KnowledgeGraphGenerator().generate(val_count, min_nodes=min_nodes, max_nodes=max_nodes, train=False)
    test_graphs = KnowledgeGraphGenerator().generate(test_count, min_nodes=min_nodes, max_nodes=max_nodes, train=False)
    return train_graphs, val_graphs, test_graphs


if __name__ == "__main__":
    train_graphs, val_graphs, test_graphs = generate_dataset(train_count=200, val_count=20, test_count=50, min_nodes=30, max_nodes=50)
    # print_example(train_graphs[0])

    print_size_stats(train_graphs)
    print_size_stats(test_graphs)

    train_label_stats = compute_label_stats(train_graphs)
    val_label_stats = compute_label_stats(val_graphs)
    test_label_stats = compute_label_stats(test_graphs)
    print('=' * 50)
    print_label_stats(train_label_stats)
    print('=' * 50)
    print_label_stats(val_label_stats)
    print('=' * 50)
    print_label_stats(test_label_stats)
