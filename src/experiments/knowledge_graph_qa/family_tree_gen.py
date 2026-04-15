import networkx as nx
import random
import json
from tqdm import tqdm

# set random seeds for reproducibility
random.seed(42)

MALE_NAMES = [
    "John", "Michael", "David", "James", "Robert", "William", "Joseph", "Charles", "Thomas", "Christopher",
    "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth",
    "Kevin", "Brian", "George", "Edward", "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob",
]

FEMALE_NAMES = [
    "Mary", "Jennifer", "Linda", "Elizabeth", "Susan", "Jessica", "Sarah", "Karen", "Nancy", "Lisa",
    "Margaret", "Betty", "Sandra", "Ashley", "Dorothy", "Kimberly", "Emily", "Donna", "Michelle",
    "Sophie", "Olivia", "Emma", "Ava", "Isabella", "Mia", "Amelia", "Harper", "Evelyn", "Abigail",
]

LAST_NAMES = [ "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin" ]

COLORS = [ "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray", "cyan", "magenta", "teal" ]
FOODS = [ "pizza", "sushi", "burger", "pasta", "salad", "steak", "tacos", "ramen", "curry", "sandwich", "cake", "chocolate", "candy", "bread", "fries" ]
CITIES = [ "Paris", "Tokyo", "London", "Sydney", "Rome", "Toronto", "Berlin", "Madrid", "Moscow", "Dubai", "Singapore", "Barcelona", "Athens", "Florence" ]

def generate_family_tree(generations=4, marriage_prob=0.8, child_prob=0.85):
    """
    Generates a realistic family tree graph, mathematically guaranteed to 
    reach the specified number of generations.
    """
    G = nx.DiGraph()
    node_counter = 0

    # Added first_name=None parameter so we can explicitly pass unique names for siblings
    def add_person(gender, birth_year, last_name, first_name=None):
        nonlocal node_counter
        node_id = node_counter
        node_counter += 1
        
        if first_name is None:
            first_name = random.choice(MALE_NAMES if gender == 'M' else FEMALE_NAMES)
        
        G.add_node(
            node_id, 
            first_name=first_name, 
            last_name=last_name, 
            gender=gender, 
            birth_year=birth_year,
            fav_color=random.choice(COLORS),
            fav_food=random.choice(FOODS),
            fav_city=random.choice(CITIES),
        )
        return node_id

    # 1. Create the founding generation
    born_year_range = (1900, 1910)
    m_born_year = random.randint(*born_year_range)
    founder_m = add_person('M', m_born_year, random.choice(LAST_NAMES))
    founder_f = add_person('F', m_born_year + random.randint(0, 5), random.choice(LAST_NAMES))

    # Add SPOUSE relation (bidirectional)
    G.add_edge(founder_m, founder_f, relation='SPOUSE')
    G.add_edge(founder_f, founder_m, relation='SPOUSE')

    # Keep track of couples that can potentially have children
    current_gen_couples = [(founder_m, founder_f)]

    # 2. Iterate through generations
    for gen in range(generations):
        next_gen_couples = []
        
        # Select a "Golden Couple" to guarantee the family line continues
        # (Only needed if we aren't on the final generation)
        golden_couple = random.choice(current_gen_couples) if gen < generations - 1 else None
        
        for husband, wife in current_gen_couples:
            is_golden_couple = ((husband, wife) == golden_couple)
            
            # Check if this couple has children (Golden Couple is guaranteed to)
            if is_golden_couple or random.random() < child_prob:
                num_children = random.randint(1, 4)
                husband_data = G.nodes[husband]
                wife_data = G.nodes[wife]

                # Track names already used by these parents to prevent duplicate sibling names
                used_first_names = set()
                
                # Pick which child will be the "Golden Child" (guaranteed to marry)
                golden_child_idx = random.randint(0, num_children - 1) if is_golden_couple else -1

                for i in range(num_children):
                    is_golden_child = (i == golden_child_idx)
                    
                    child_gender = random.choice(['M', 'F'])
                    # Child inherits father's last name
                    child_last_name = husband_data['last_name']
                    # Child is born 20-35 years after the mother
                    child_birth_year = wife_data['birth_year'] + random.randint(20, 35)

                    # Ensure the child gets a unique first name among their siblings
                    while True:
                        child_first_name = random.choice(MALE_NAMES if child_gender == 'M' else FEMALE_NAMES)
                        if child_first_name not in used_first_names:
                            used_first_names.add(child_first_name)
                            break

                    child_id = add_person(child_gender, child_birth_year, child_last_name, first_name=child_first_name)

                    # Add CHILD relations (directed from parent to child)
                    G.add_edge(husband, child_id, relation='CHILD')
                    G.add_edge(wife, child_id, relation='CHILD')

                    # Check if the child gets married 
                    # (Golden Child is guaranteed to marry to continue the line)
                    if gen < generations - 1 and (is_golden_child or random.random() < marriage_prob):
                        # Ensure heterosexual marriage
                        spouse_gender = 'F' if child_gender == 'M' else 'M'
                        
                        # Spouse gets a random last name (representing their own family origin)
                        spouse_last_name = random.choice(LAST_NAMES)
                        # Spouse is roughly the same age (-5 to +5 years)
                        spouse_birth_year = child_birth_year + random.randint(-5, 5)
                        
                        spouse_id = add_person(spouse_gender, spouse_birth_year, spouse_last_name)

                        # Add SPOUSE relation (bidirectional)
                        G.add_edge(child_id, spouse_id, relation='SPOUSE')
                        G.add_edge(spouse_id, child_id, relation='SPOUSE')

                        # Add the new couple to the list for the next generation
                        if child_gender == 'M':
                            next_gen_couples.append((child_id, spouse_id))
                        else:
                            next_gen_couples.append((spouse_id, child_id))

        current_gen_couples = next_gen_couples

    return G

def generate_qa_pair(G):
    """
    Generates a unique Q&A pair based on the family tree graph.
    Returns: (person_id, question, answer)
    """
    def get_ordinal(n):
        if 11 <= (n % 100) <= 13:
            return f"{n}th"
        return f"{n}" + ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]

    # 1. Collect all unique names to avoid anchor ambiguity 
    # (e.g. if there are two 'David Smiths', asking about David Smith is ambiguous)
    name_counts = {}
    for n, d in G.nodes(data=True):
        name = f"{d['first_name']} {d['last_name']}"
        name_counts[name] = name_counts.get(name, 0) + 1
        
    valid_anchors = [n for n, d in G.nodes(data=True) if name_counts[f"{d['first_name']} {d['last_name']}"] == 1]
    
    # Randomize the traversal so we get a random person and random question each time
    random.shuffle(valid_anchors)
    
    for u in valid_anchors:
        # 2. Gather all relatives of node 'u'
        parents = [v for v, _, d in G.in_edges(u, data=True) if d['relation'] == 'CHILD']
        children = [v for _, v, d in G.out_edges(u, data=True) if d['relation'] == 'CHILD']
        spouses = [v for _, v, d in G.out_edges(u, data=True) if d['relation'] == 'SPOUSE']
        
        # Deduplicate
        parents = list(set(parents))
        children = list(set(children))
        spouses = list(set(spouses))
        
        # Extended family (2-steps)
        siblings = []
        for p in parents:
            for p_child in [v for _, v, d in G.out_edges(p, data=True) if d['relation'] == 'CHILD']:
                if p_child != u:
                    siblings.append(p_child)
        siblings = list(set(siblings))
        
        grandparents = []
        for p in parents:
            grandparents.extend([v for v, _, d in G.in_edges(p, data=True) if d['relation'] == 'CHILD'])
        grandparents = list(set(grandparents))
        
        grandchildren = []
        for c in children:
            grandchildren.extend([v for _, v, d in G.out_edges(c, data=True) if d['relation'] == 'CHILD'])
        grandchildren = list(set(grandchildren))
        
        uncles_aunts = []
        for p in parents:
            p_parents = [v for v, _, d in G.in_edges(p, data=True) if d['relation'] == 'CHILD']
            for pp in p_parents:
                for pp_child in [v for _, v, d in G.out_edges(pp, data=True) if d['relation'] == 'CHILD']:
                    if pp_child != p:
                        uncles_aunts.append(pp_child)
        uncles_aunts = list(set(uncles_aunts))
        
        # 3. Categorize relationships by gender
        relations = {
            "father": [v for v in parents if G.nodes[v]['gender'] == 'M'],
            "mother": [v for v in parents if G.nodes[v]['gender'] == 'F'],
            "husband": [v for v in spouses if G.nodes[v]['gender'] == 'M'],
            "wife": [v for v in spouses if G.nodes[v]['gender'] == 'F'],
            "son": [v for v in children if G.nodes[v]['gender'] == 'M'],
            "daughter": [v for v in children if G.nodes[v]['gender'] == 'F'],
            "brother": [v for v in siblings if G.nodes[v]['gender'] == 'M'],
            "sister": [v for v in siblings if G.nodes[v]['gender'] == 'F'],
            "grandfather": [v for v in grandparents if G.nodes[v]['gender'] == 'M'],
            "grandmother": [v for v in grandparents if G.nodes[v]['gender'] == 'F'],
            "grandson": [v for v in grandchildren if G.nodes[v]['gender'] == 'M'],
            "granddaughter": [v for v in grandchildren if G.nodes[v]['gender'] == 'F'],
            "uncle": [v for v in uncles_aunts if G.nodes[v]['gender'] == 'M'],
            "aunt": [v for v in uncles_aunts if G.nodes[v]['gender'] == 'F'],
        }
        
        # Remove empty categories and shuffle
        relations = {k: v for k, v in relations.items() if len(v) > 0}
        rel_items = list(relations.items())
        random.shuffle(rel_items)
        
        # 4. Find a valid question
        for rel_name, targets in rel_items:
            # Sort by birth year to establish oldest/youngest logic
            targets.sort(key=lambda x: G.nodes[x]['birth_year'])
            
            # Verify there are no duplicate birth years among this group to prevent ambiguity
            birth_years = [G.nodes[t]['birth_year'] for t in targets]
            if len(birth_years) != len(set(birth_years)):
                continue # Skip this relation type for this person if ambiguous (e.g., twins)
                
            target = random.choice(targets)
            target_idx = targets.index(target)
            
            # Format relation descriptor
            if len(targets) == 1:
                rel_desc = rel_name
            else:
                k = target_idx + 1
                rel_desc = f"{get_ordinal(k)} oldest {rel_name}"
                
            prop = random.choice(["color", "food", "city"])
            
            anchor_data = G.nodes[u]
            anchor_name = f"{anchor_data['first_name']} {anchor_data['last_name']}"
            
            question = f"What is the favourite {prop} of {anchor_name}'s {rel_desc}?"
            answer = G.nodes[target][f"fav_{prop}"]
            
            return u, question, answer
            
    return None, "Could not generate an unambiguous question.", None

def export_tree_to_json(G, filename="family_tree.json"):
    """Exports the networkx graph to a JSON file."""
    # Convert graph data into a dictionary suitable for JSON serialization
    data = nx.node_link_data(G)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print(f"\nFamily tree successfully exported to {filename}")

def print_family_tree(G):
    """
    Prints a networkx family tree in a hierarchical ASCII format.
    """
    visited = set()
    
    # Added an 'is_root' parameter to correctly identify the top level
    def print_person(node, prefix, is_last, is_root=False):
        if node in visited:
            return
        
        # Mark as visited
        visited.add(node)
        
        # Get person data
        data = G.nodes[node]
        info = f"{data['first_name']} {data['last_name']} ({data['gender']}, b. {data['birth_year']}; {data['fav_color']}; {data['fav_food']}; {data['fav_city']})"
        
        # 1. Identify spouse(s)
        spouses = [v for _, v, d in G.out_edges(node, data=True) if d['relation'] == 'SPOUSE' and v not in visited]
        spouse_str = ""
        for sp in spouses:
            visited.add(sp) # Mark spouse as visited so they don't get printed on a separate line
            s_data = G.nodes[sp]
            spouse_str += f" <=> {s_data['first_name']} {s_data['last_name']} ({s_data['gender']}, b. {s_data['birth_year']}; {s_data['fav_color']}; {s_data['fav_food']}; {s_data['fav_city']})"
            
        # 2. Print current node + spouse
        marker = "└── " if is_last else "├── "
        
        if is_root: 
            print(f"{info}{spouse_str}")
            child_prefix = ""
        else:
            print(f"{prefix}{marker}{info}{spouse_str}")
            # Build the prefix for the next generation
            child_prefix = prefix + ("    " if is_last else "│   ")
            
        # 3. Gather children
        children = [v for _, v, d in G.out_edges(node, data=True) if d['relation'] == 'CHILD' and v not in visited]
        for sp in spouses:
            sp_children = [v for _, v, d in G.out_edges(sp, data=True) if d['relation'] == 'CHILD' and v not in visited]
            children.extend(sp_children)
            
        # Remove duplicates and sort by birth year
        children = list(set(children))
        children.sort(key=lambda x: G.nodes[x]['birth_year'])
        
        # 4. Recursively print children
        for i, child in enumerate(children):
            # Pass is_root=False to all children
            print_person(child, child_prefix, i == len(children) - 1, is_root=False)

    # Find the "founders" of the tree
    founders = []
    for n in G.nodes():
        incoming_child_edges = [u for u, v, d in G.in_edges(n, data=True) if d['relation'] == 'CHILD']
        if not incoming_child_edges:
            founders.append(n)
    
    print("\n" + "="*50)
    print(f"FAMILY TREE ({G.number_of_nodes()} people)")
    print("="*50)
    # print each persons id and their full name
    for n in G.nodes():
        data = G.nodes[n]
        print(f"{n}: {data['first_name']} {data['last_name']}")
    print("="*50)
    
    # Sort founders by birth year and print them
    for f in sorted(founders, key=lambda x: G.nodes[x]['birth_year']):
        if f not in visited:
            # Pass is_root=True for the founders
            print_person(f, "", True, is_root=True)
            print()


def generate_dataset(n_train=700, n_val=100, n_test=200, generations=3, marriage_prob=0.7, child_prob=0.75, return_dict=False):
    """
    Generates a datasets of family trees and Q/A pairs.
    """
    datasets = {"train": [], "val": [], "test": []}
    
    for split, n_samples in zip(["train", "val", "test"], [n_train, n_val, n_test]):
        print(f"\nGenerating {split} set with {n_samples} samples...")
        for _ in tqdm(range(n_samples)):
            G = generate_family_tree(generations=generations, marriage_prob=marriage_prob, child_prob=child_prob)
            person_id, question, answer = generate_qa_pair(G)
            datasets[split].append({
                "graph": G,
                "person_id": person_id,
                "question": question,
                "answer": answer,
            })
    
    if return_dict:
        return datasets
    else:
        return datasets["train"], datasets["val"], datasets["test"]


if __name__ == "__main__":

    train_dataset, val_dataset, test_dataset = generate_dataset(n_train=700, n_val=100, n_test=200, generations=3, marriage_prob=0.7, child_prob=0.75)

    total_people = 0
    total_trials = 0

    for sample in train_dataset:
        total_people += sample["graph"].number_of_nodes()
        total_trials += 1

    print(f"\nAverage people per trial: {total_people / total_trials}")
    print('-'*50)

    # Print the family tree in a readable format
    family_tree = train_dataset[0]["graph"]
    print_family_tree(family_tree)

    print("\nGenerating a Q/A pair about a relative's favourite attribute...")
    print("-"*50)
    qa_pair = generate_qa_pair(family_tree)
    if qa_pair:
        person_id, question, answer = qa_pair
        print(f"Person ID: {person_id}")
        print(f"\nQuestion: {question}")
        print(f"Answer: {answer}")
    else:
        raise ValueError("Failed to generate a valid Q/A pair from the family tree.")
    print("-"*50)
    