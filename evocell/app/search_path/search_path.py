import pandas as pd
import networkx as nx

def filter_paths_by_length(valid_paths, current_valid_paths, keep_longest=True):
    # Combine valid_paths and current_valid_paths into a single list of tuples
    combined_paths = list(zip(valid_paths, current_valid_paths))
    
    # Sort the combined paths by the length of the paths in current_valid_paths
    combined_paths.sort(key=lambda x: len(x[1]), reverse=keep_longest)

    filtered_valid_paths = []
    filtered_current_paths = []

    for valid_path, current_path in combined_paths:
        if keep_longest:
            # Keep path if it's not contained within any already filtered path
            if not any(set(current_path).issubset(set(existing_path)) for existing_path in filtered_current_paths):
                filtered_valid_paths.append(valid_path)
                filtered_current_paths.append(current_path)
        else:
            # Shortest option: Add the path only if it is not a subset of any path already in filtered_current_paths
            if not any(set(existing_path).issubset(set(current_path)) for existing_path in filtered_current_paths):
                filtered_valid_paths.append(valid_path)
                filtered_current_paths.append(current_path)

    return filtered_valid_paths, filtered_current_paths

def filter_paths_by_length_old(paths, keep_longest=True):
    # Sort the paths by length based on the keep_longest option
    paths.sort(key=len, reverse=keep_longest)

    filtered_paths = []
    for path in paths:
        if keep_longest:
            # Keep path if it's not contained within any already filtered path
            if not any(set(path).issubset(set(existing_path)) for existing_path in filtered_paths):
                 filtered_paths.append(path)
        else:
            # Shortest option: Add the path only if it is not a subset of any path already in filtered_paths
            if not any(set(existing_path).issubset(set(path)) for existing_path in filtered_paths):
                 filtered_paths.append(path)

    return filtered_paths

def analyze_subgraphs(cyclic_trajectory_edges, milestones_metadata, verbose=False):
    # Create the graph
    G = nx.Graph()
    
    # Add edges from cyclic_trajectory_edges DataFrame
    for _, row in cyclic_trajectory_edges.iterrows():
        G.add_edge(row['from'], row['to'], length=row['length'])
    
    # Get connected components (subgraphs) of G
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    # Prepare to collect data about subgraphs
    subgraph_info = []

    # Analyze each subgraph
    for index, subgraph in enumerate(subgraphs):
        # Number of milestones (nodes)
        num_milestones = len(subgraph.nodes)

        # Check if subgraph is cyclic
        is_cyclic = not nx.is_tree(subgraph)

        # Extract milestones and their cell types
        milestones_in_subgraph = list(subgraph.nodes)
        #print(milestones_in_subgraph)
        #print(milestones_metadata['milestone_id'].isin(milestones_in_subgraph))
        
        milestones_data = milestones_metadata[milestones_metadata['milestone_id'].isin(milestones_in_subgraph)]
        #print(milestones_data)
        # Count cell types associated with the milestones
        celltype_counts = milestones_data['celltype'].value_counts().to_dict()
        
        # Add information to results list
        subgraph_info.append({
            'subgraph_index': index + 1,
            'num_milestones': num_milestones,
            'is_cyclic': is_cyclic,
            'celltype_distribution': celltype_counts
        })
        
        if verbose:
            print(f"Subgraph {index + 1}:")
            print(f"  - Number of milestones: {num_milestones}")
            print(f"  - Is cyclic: {is_cyclic}")
            print(f"  - Celltype distribution: {celltype_counts}")
            print("")

    # Create a DataFrame for display
    subgraph_df = pd.DataFrame(subgraph_info)
    return subgraph_df

def get_milestones_paths(metadata_col, cyclic_trajectory_edges, milestones_metadata, stage_order, verbose=False):
    # Create the graph from cyclic_trajectory_edges
    G = nx.Graph()
    for _, row in cyclic_trajectory_edges.iterrows():
        G.add_edge(row['from'], row['to'], length=row['length'])
    
    # Get all connected components (subgraphs) of G
    subgraphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    path_infos = []
    current_paths = []
    
    # Analyze each connected subgraph
    for subgraph_index, subgraph in enumerate(subgraphs):
        if verbose:
            print(f"Analyzing subgraph {subgraph_index + 1} with nodes {list(subgraph.nodes)}")
        
        # Find milestones in the current subgraph
        milestones_in_subgraph = list(subgraph.nodes)
        milestones_data_subgraph = milestones_metadata[milestones_metadata['milestone_id'].isin(milestones_in_subgraph)]
        
        # Get nodes in the subgraph that match the current stage order
        start_stage, end_stage = stage_order  # Unpack the two stages
        start_milestones = milestones_data_subgraph[milestones_data_subgraph[metadata_col] == start_stage]['milestone_id'].tolist()
        end_milestones = milestones_data_subgraph[milestones_data_subgraph[metadata_col] == end_stage]['milestone_id'].tolist()
        
        if verbose:
            print(f"  Start stage ({start_stage}): {start_milestones}")
            print(f"  End stage ({end_stage}): {end_milestones}")
        
        # Find paths between start and end milestones
        for start_node in start_milestones:
            for end_node in end_milestones:
                try:
                    # Find the shortest path in the subgraph
                    path = nx.shortest_path(subgraph, source=start_node, target=end_node, weight='length')
                    #path_stages = milestones_metadata[milestones_metadata['milestone_id'].isin(path)][metadata_col].tolist()
                    # Assuming milestones_metadata is a DataFrame with 'milestone_id' as a column and metadata_col as the column of interest
                    path_stages = [milestones_metadata.loc[milestones_metadata['milestone_id'] == node, metadata_col].values[0] for node in path]


                    # For debugging: print the path found
                    if verbose:
                        print(f"    Path found from {start_node} to {end_node}: {path} with stages {path_stages}")
                    
                    # Store the path
                    path_infos.append({
                        "subgraph_index": subgraph_index + 1,
                        "node": path,
                        "stage": path_stages,
                        "start_stage": start_stage,
                        "end_stage": end_stage
                    })
                    current_paths.append(path)
                    
                except nx.NetworkXNoPath:
                    if verbose:
                        print(f"    No path found between {start_node} and {end_node}")
                    continue
    
    return path_infos, current_paths

def filter_paths_by_neighbors_count(path_infos, current_paths, neighbors_count, verbose=False):
    valid_paths = []
    current_valid_paths = []

    for path_info, current_path in zip(path_infos, current_paths):
        path = path_info["node"]
        stages = path_info["stage"]

        # If neighbors_count is 0, keep all paths
        if neighbors_count == 0:
            valid_paths.append(path_info)
            current_valid_paths.append(current_path)
            continue

        # Check if the path is long enough to validate both ends
        required_length = 2 + 2 * neighbors_count
        if len(stages) < required_length:
            if verbose:
                print(f"Path {path} is too short to validate {neighbors_count} neighbors on both ends.")
            continue
        
        # Check if the first neighbors_count nodes share the same stage as the start node
        start_stage = stages[0]
        start_neighbors = stages[1:1 + neighbors_count]  # Neighbors of the first node
        if not all(stage == start_stage for stage in start_neighbors):
            if verbose:
                print(f"Path {path} filtered out due to differing stages in the first {neighbors_count} nodes: {start_neighbors}")
            continue
        
        # Check if the last neighbors_count nodes share the same stage as the end node
        end_stage = stages[-1]
        end_neighbors = stages[-(1 + neighbors_count):-1]  # Neighbors of the last node
        if not all(stage == end_stage for stage in end_neighbors):
            if verbose:
                print(f"Path {path} filtered out due to differing stages in the last {neighbors_count} nodes: {end_neighbors}")
            continue
        
        # If both the start and end of the path are valid, add the path to the valid paths list
        if verbose:
            print(f"Keeping Path {path}")
            
        valid_paths.append(path_info)
        current_valid_paths.append(current_path)
    
    # Return the filtered path_infos and the filtered current_paths
    return valid_paths, current_valid_paths

def filter_paths_by_neighbors_count_old(path_infos, current_paths, neighbors_count, verbose=False):
    valid_paths = []

    for path_info in path_infos:
        path = path_info["node"]
        stages = path_info["stage"]

        # If neighbors_count is 0, keep all paths
        if neighbors_count == 0:
            valid_paths.append(path_info)
            continue

        # Check if the path is long enough to validate both ends
        required_length = 2 + 2 * neighbors_count
        if len(stages) < required_length:
            if verbose:
                #print(f"Path {path} is too short to validate {neighbors_count} neighbors on both ends.")
                continue
        
        # Check if the first neighbors_count nodes share the same stage as the start node
        start_stage = stages[0]
        start_neighbors = stages[1:1 + neighbors_count]  # Neighbors of the first node
        if not all(stage == start_stage for stage in start_neighbors):
            if verbose:
                #print(f"Path {path} filtered out due to differing stages in the first {neighbors_count} nodes: {start_neighbors}")
                continue
        
        # Check if the last neighbors_count nodes share the same stage as the end node
        end_stage = stages[-1]
        end_neighbors = stages[-(1 + neighbors_count):-1]  # Neighbors of the last node
        if not all(stage == end_stage for stage in end_neighbors):
            if verbose:
                #print(f"Path {path} filtered out due to differing stages in the last {neighbors_count} nodes: {end_neighbors}")
                continue
        
        # If both the start and end of the path are valid, add the path to the valid paths list

        if verbose:
            print(f"Keeping Path {path}")
            
        valid_paths.append(path_info)
    
    return valid_paths

def assign_cells_to_milestones(cell_data, milestone_percentages, celltype_col, time_col, cell_id_col):
    # Step 1: Select the row with the maximum percentage for each cell_id
    cell_id_milestones_df = milestone_percentages.loc[
        milestone_percentages.groupby('cell_id')['percentage'].idxmax()
    ]

    # Step 2: Merge cell_data with cell_id_milestones_df on cell_id
    merged_df = pd.merge(cell_data, cell_id_milestones_df, left_on=cell_id_col, right_on="cell_id")

    # Step 3: Calculate proportions for each (milestone_id, celltype, time) combination
    proportions_df = (
        merged_df
        .groupby(['milestone_id', celltype_col, time_col])
        .size()
        .reset_index(name='count')
    )
    
    
    proportions_df['total'] = proportions_df.groupby('milestone_id')['count'].transform('sum')
    proportions_df['proportion'] = proportions_df['count'] / proportions_df['total']

    # Drop unnecessary columns
    proportions_df = proportions_df.drop(columns=['count', 'total'])

    # Step 4: Rank the proportions within each milestone_id and select the top 4
    proportions_df['rank'] = proportions_df.groupby('milestone_id')['proportion'].rank(method='first', ascending=False)
    top4_df = proportions_df[proportions_df['rank'] <= 4]

    # Step 5: Arrange and rename as necessary
    milestones_top4_metadata = (
        top4_df
        .assign(rank=lambda df: 'top' + df['rank'].astype(int).astype(str))
        .sort_values(by=['milestone_id', 'rank'], ascending=[True, True])
        .loc[:, ['milestone_id', 'rank', celltype_col, time_col, 'proportion']]
    )

    return milestones_top4_metadata

def assign_cells_to_milestones_no_time(cell_data, milestone_percentages, celltype_col, cell_id_col):
    # Step 1: Select the row with the maximum percentage for each cell_id
    cell_id_milestones_df = milestone_percentages.loc[
        milestone_percentages.groupby('cell_id')['percentage'].idxmax()
    ]

    # Step 2: Merge cell_data with cell_id_milestones_df on cell_id
    merged_df = pd.merge(cell_data, cell_id_milestones_df, left_on=cell_id_col, right_on="cell_id")

    # Step 3: Calculate proportions for each (milestone_id, celltype) combination
    proportions_df = (
        merged_df
        .groupby(['milestone_id', celltype_col])
        .size()
        .reset_index(name='count')
    )

    print(proportions_df)
    print(proportions_df[proportions_df["milestone_id"]=="M341"])
    
    proportions_df['total'] = proportions_df.groupby('milestone_id')['count'].transform('sum')
    proportions_df['proportion'] = proportions_df['count'] / proportions_df['total']

    # Drop unnecessary columns
    proportions_df = proportions_df.drop(columns=['count', 'total'])

    # Step 4: Rank the proportions within each milestone_id and select the top 4
    proportions_df['rank'] = proportions_df.groupby('milestone_id')['proportion'].rank(method='first', ascending=False)
    top4_df = proportions_df[proportions_df['rank'] <= 4]

    # Step 5: Arrange and rename as necessary
    milestones_top4_metadata = (
        top4_df
        .assign(rank=lambda df: 'top' + df['rank'].astype(int).astype(str))
        .sort_values(by=['milestone_id', 'rank'], ascending=[True, True])
        .loc[:, ['milestone_id', 'rank', celltype_col, 'proportion']]
    )

    return milestones_top4_metadata

def find_leaf_node_paths(trajectory_edges):
    """
    Given a DataFrame of edges, create a graph and find all shortest paths between pairs of leaf nodes.

    Parameters:
    trajectory_edges (pandas.DataFrame): A DataFrame containing 'source' and 'target' columns for edges.

    Returns:
    dict: A dictionary containing all shortest paths between pairs of leaf nodes, 
          where keys are pairs of leaf nodes and values are the shortest path vectors.
    """
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add edges to the graph
    for index, row in trajectory_edges.iterrows():
        G.add_edge(row['from'], row['to'])

    # Convert to an undirected graph
    G_undirected = G.to_undirected()

    # Identify leaf nodes (nodes with degree 1 in the undirected graph)
    leaf_nodes = [node for node in G_undirected.nodes if G_undirected.degree(node) == 1]

    # Dictionary to store all paths between pairs of leaf nodes
    all_paths_between_leaves_as_vectors = {}

    # Loop over each pair of leaf nodes and find the shortest paths
    for i in range(len(leaf_nodes) - 1):
        for j in range(i + 1, len(leaf_nodes)):
            try:
                # Find all shortest paths between the leaf node pair in the undirected graph
                paths = list(nx.all_shortest_paths(G_undirected, source=leaf_nodes[i], target=leaf_nodes[j]))
                # Check if any path is found
                if paths:
                    # Store the path in the dictionary
                    path_as_vector = paths[0]  # Assuming taking the first path if multiple are found
                    all_paths_between_leaves_as_vectors[f"{leaf_nodes[i]}-{leaf_nodes[j]}"] = path_as_vector
            except nx.NetworkXNoPath:
                # Handle cases where no path exists between the nodes
                pass

    return all_paths_between_leaves_as_vectors


def get_root_node_end_node(all_paths, milestones_metadata, stage_order, time_order, neighbors_count, strict_time_difference=False):
    def find_direct_path(milestones, stages, times, stage_order, time_order, neighbors_count, strict_time_difference=False):
        print("find_direct_path")
        
        # Map stages to ordered categorical values
        stages = pd.Categorical(stages, categories=stage_order, ordered=True)

        print(times)
        # Map times to their order indices in time_order
        time_indices = [time_order.index(t) for t in times]

        start_indices = [i for i, s in enumerate(stages) if s == stage_order[0]]
        end_indices = [i for i, s in enumerate(stages) if s == stage_order[1]]

        print(f"start_indices: {start_indices}")
        print(f"end_indices: {end_indices}")

        def check_neighbors(index, time_indices, stages, neighbors_count):
            # Generate indices for neighbors before and after, within bounds
            neighbors_indices = range(max(0, index - neighbors_count), min(len(time_indices), index + neighbors_count + 1))
            neighbors_indices = [ni for ni in neighbors_indices if ni != index]  # Exclude the current index
            
            # Check if all neighbors have the same time index and stage
            all_same_time_stage = all(time_indices[ni] == time_indices[index] for ni in neighbors_indices) and \
                                  all(stages[ni] == stages[index] for ni in neighbors_indices)
            
            return all_same_time_stage
        
        path_info = pd.DataFrame(columns=["node", "stage", "time"])
        
        if len(end_indices) > 0:
            for start_index in start_indices:
                path_found = False

                if not check_neighbors(start_index, time_indices, stages, neighbors_count):
                    continue

                for end_index in end_indices:
                    if not check_neighbors(end_index, time_indices, stages, neighbors_count):
                        continue
                    
                    # Now, using time_indices for comparison
                    if (strict_time_difference and time_indices[start_index] < time_indices[end_index]) or \
                       (not strict_time_difference and time_indices[start_index] <= time_indices[end_index]):
                        # Found a valid path
                        path_info = path_info.append({
                            "node": [milestones[start_index], milestones[end_index]],
                            "stage": [stages[start_index], stages[end_index]],
                            "time": [times[start_index], times[end_index]]
                        }, ignore_index=True)
                        path_found = True
                        break
                
                if path_found:
                    break

        if not path_info.empty:
            print(path_info)
            return path_info
        else:
            print("path not found")
            return None

    def reorder_and_subset_path(current_path, root_end_node_ordered_vector):
        print("reorder_and_subset_path")
        root_node, end_node = root_end_node_ordered_vector
        root_node_position = current_path.index(root_node)
        end_node_position = current_path.index(end_node)
        
        start_position = min(root_node_position, end_node_position)
        end_position = max(root_node_position, end_node_position)
        
        subsetted_path = current_path[start_position:end_position+1]
        
        if root_node_position > end_node_position:
            subsetted_path = list(reversed(subsetted_path))
        
        return subsetted_path
    
    path_infos = []
    current_paths = []
    
    for path_name, current_path in all_paths.items():
        print(f"Vertices1: {current_path}")

        #TODO celltype as parameter
        development_stages = milestones_metadata.loc[
            milestones_metadata['milestone_id'].isin(current_path), 'celltype'
        ].tolist()

        #TODO day as parameter
        times = milestones_metadata.loc[
            milestones_metadata['milestone_id'].isin(current_path), 'day'
        ].tolist()

        path_info = find_direct_path(current_path, development_stages, times, stage_order, time_order, neighbors_count, strict_time_difference)

        if path_info is not None:
            print(f"Path: {path_name}")
            path_infos.append(path_info)
            current_paths.append(reorder_and_subset_path(current_path, path_info['node'].tolist()[0]))

    return path_infos, current_paths



def get_root_node_end_node_without_time(all_paths, milestones_metadata, stage_order, neighbors_count, verbose=False):
    def find_direct_path(milestones, stages, stage_order, neighbors_count):
        if verbose:
            print("Entering find_direct_path")
            print(f"Milestones: {milestones}")
            print(f"Stages: {stages}")
            print(f"Stage order: {stage_order}")
        
        # Map stages to ordered categorical values
        stages = pd.Categorical(stages, categories=stage_order, ordered=True)
        if verbose:
            print(f"Ordered stages: {stages}")
        
        start_indices = [i for i, s in enumerate(stages) if s == stage_order[0]]
        end_indices = [i for i, s in enumerate(stages) if s == stage_order[1]]

        if verbose:
            print(f"Start indices: {start_indices}")
            print(f"End indices: {end_indices}")

        def check_neighbors(index, stages, neighbors_count):
            if verbose:
                print(f"Checking neighbors for index: {index}, stage: {stages[index]}")
            # Generate indices for neighbors before and after, within bounds
            #neighbors_indices = range(max(0, index - neighbors_count), min(len(stages), index + neighbors_count + 1))
            neighbors_indices = range(max(0, index - neighbors_count), min(len(stages), index + neighbors_count))
            neighbors_indices = [ni for ni in neighbors_indices if ni != index]  # Exclude the current index
            
            # Check if all neighbors have the same stage
            all_same_stage = all(stages[ni] == stages[index] for ni in neighbors_indices)
            
            if verbose:
                print(f"Neighbors indices: {list(neighbors_indices)}")
                print(f"All neighbors have the same stage: {all_same_stage}")
            
            return all_same_stage
        
        path_info = pd.DataFrame(columns=["node", "stage"])
        
        if len(end_indices) > 0:
            for start_index in start_indices:
                path_found = False
                if verbose:
                    print(f"Evaluating start index: {start_index}")

                if not check_neighbors(start_index, stages, neighbors_count):
                    if verbose:
                        print(f"Neighbors condition not met for start index: {start_index}")
                    continue

                for end_index in end_indices:
                    if verbose:
                        print(f"Evaluating end index: {end_index}")
                    if not check_neighbors(end_index, stages, neighbors_count):
                        if verbose:
                            print(f"Neighbors condition not met for end index: {end_index}")
                        continue
                    
                    # Found a valid path without considering time
                    if verbose:
                        print(f"Valid path found between milestones: {milestones[start_index]} and {milestones[end_index]}")
                    path_info = pd.concat([
                        path_info,
                        pd.DataFrame({
                            "node": [[milestones[start_index], milestones[end_index]]],
                            "stage": [[stages[start_index], stages[end_index]]],
                        })
                    ], ignore_index=True)
                    path_found = True
                    break
                
                if path_found:
                    break

        if not path_info.empty:
            if verbose:
                print("Path info found:")
                print(path_info)
            return path_info
        else:
            if verbose:
                print("No valid path found")
            return None

    def reorder_and_subset_path(current_path, root_end_node_ordered_vector):
        if verbose:
            print("Reordering and subsetting path")
            print(f"Current path: {current_path}")
            print(f"Root and end nodes: {root_end_node_ordered_vector}")
        
        root_node, end_node = root_end_node_ordered_vector
        root_node_position = current_path.index(root_node)
        end_node_position = current_path.index(end_node)
        
        start_position = min(root_node_position, end_node_position)
        end_position = max(root_node_position, end_node_position)
        
        subsetted_path = current_path[start_position:end_position+1]
        
        if root_node_position > end_node_position:
            subsetted_path = list(reversed(subsetted_path))
        
        if verbose:
            print(f"Subsetted path: {subsetted_path}")
        return subsetted_path
    
    path_infos = []
    current_paths = []
    
    #for path_name, current_path in all_paths.items():
    for current_path in all_paths:
        # Process each path
        if verbose:
            #print(f"Processing path: {path_name}")
            print(f"Current path vertices: {current_path}")

        # TODO: celltype as parameter
        development_stages = milestones_metadata.loc[
            milestones_metadata['milestone_id'].isin(current_path), 'celltype'
        ].tolist()
        if verbose:
            #print(f"Development stages for path {path_name}: {development_stages}")
            print(f"Development stages for {development_stages}")
        path_info = find_direct_path(current_path, development_stages, stage_order, neighbors_count)

        if path_info is not None:
            if verbose:
                #print(f"Path found for {path_name}: {path_info}")
                print(f"Path found for {path_info}")
                
            path_infos.append(path_info)
            current_paths.append(reorder_and_subset_path(current_path, path_info['node'].tolist()[0]))
        else:
            if verbose:
                #print(f"No path found for {path_name}")
                print(f"No path found")

    return path_infos, current_paths

def reconstruct_paths_working(dimred_milestone2, verbose=False):
    def simplify_graph(graph):
        # Dictionary to store the full path for each simplified edge
        full_path_correspondence = {}
        
        # Create a copy of the graph to modify
        G_simplified = graph.copy()
        
        # Continue simplifying while there are nodes with degree 2
        degree_2_nodes = [node for node in G_simplified.nodes() if G_simplified.degree(node) == 2]
        
        if verbose:
            print("Initial degree 2 nodes:", degree_2_nodes)
        
        while degree_2_nodes:
            # Take the first node with degree 2
            node = degree_2_nodes.pop()
            
            if verbose:
                print(f"Simplifying node: {node}")
            
            # Get its neighbors
            neighbors = list(G_simplified.neighbors(node))
            if len(neighbors) != 2:
                continue  # Safety check, should always be true for degree 2 nodes
            
            n1, n2 = neighbors
            
            # Create or extend the full path correspondence
            if (n1, n2) in full_path_correspondence:
                full_path_correspondence[(n1, n2)].insert(-1, node)
            elif (n2, n1) in full_path_correspondence:
                full_path_correspondence[(n2, n1)].insert(1, node)
            else:
                full_path_correspondence[(n1, n2)] = [n1, node, n2]
            
            # Remove the node from the graph
            G_simplified.remove_node(node)
            
            # Add an edge directly between the neighbors (n1 and n2)
            if not G_simplified.has_edge(n1, n2):
                G_simplified.add_edge(n1, n2)
            
            # Recalculate nodes with degree 2 in case simplification introduces new ones
            degree_2_nodes = [node for node in G_simplified.nodes() if G_simplified.degree(node) == 2]
            
            if verbose:
                print(f"Updated degree 2 nodes: {degree_2_nodes}")

        # Simplify the full path correspondence by merging nodes correctly
        for key in full_path_correspondence:
            unique_path = []
            for node in full_path_correspondence[key]:
                if node not in unique_path:
                    unique_path.append(node)
            full_path_correspondence[key] = unique_path
        
        if verbose:
            print("Final full path correspondence:", full_path_correspondence)

        return G_simplified, full_path_correspondence

    def find_all_paths(graph):
        all_paths_dict = {}
        nodes = list(graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                source, target = nodes[i], nodes[j]
                all_paths = list(nx.all_simple_paths(graph, source, target))
                for path in all_paths:
                    key = f"{path[0]}-{path[-1]}"
                    all_paths_dict[key] = path
        
        if verbose:
            print("All paths found in the graph:", all_paths_dict)
        
        return all_paths_dict

    def remove_subpaths(all_paths_dict):
        all_paths = list(all_paths_dict.values())
        filtered_paths = set()
        for path in all_paths:
            if not any(set(path).issubset(set(other_path)) and path != other_path for other_path in all_paths):
                filtered_paths.add(tuple(path))
        
        filtered_paths_dict = {f"{path[0]}-{path[-1]}": list(path) for path in filtered_paths}
        
        if verbose:
            print("Filtered paths after removing sub-paths:", filtered_paths_dict)
        
        return filtered_paths_dict

    def expand_full_path(start_node, end_node, full_path_corr):
        """Recursively expand the full path between two nodes."""
        if (start_node, end_node) in full_path_corr:
            segment = full_path_corr[(start_node, end_node)]
        elif (end_node, start_node) in full_path_corr:
            segment = full_path_corr[(end_node, start_node)][::-1]
        else:
            return [start_node, end_node]
        
        # Recursively expand each segment in the full path
        full_expanded_path = [segment[0]]
        for i in range(len(segment) - 1):
            start, end = segment[i], segment[i + 1]
            full_expanded_path.extend(expand_full_path(start, end, full_path_corr)[1:])
        
        return full_expanded_path

    def reconstruct_full_paths(filtered_paths_dict, full_path_corr):
        reconstructed_paths = {}
        for key, path in filtered_paths_dict.items():
            full_path = []
            for i in range(len(path) - 1):
                start_node = path[i]
                end_node = path[i + 1]
                
                # Expand the full path between consecutive nodes
                expanded_segment = expand_full_path(start_node, end_node, full_path_corr)
                
                # Add the expanded segment to the full path
                if full_path:
                    full_path.extend(expanded_segment[1:])
                else:
                    full_path.extend(expanded_segment)
            
            reconstructed_key = f"{full_path[0]}-{full_path[-1]}"
            reconstructed_paths[reconstructed_key] = full_path
        
        if verbose:
            print("Reconstructed full paths:", reconstructed_paths)
        
        return reconstructed_paths

    # Create the graph from the given edges
    G = nx.Graph()
    for _, row in dimred_milestone2.iterrows():
        G.add_edge(row['from'], row['to'], length=row['length'])
    
    if verbose:
        print("Initial graph created with edges:", G.edges(data=True))
    
    # Initialize the final reconstructed paths dictionary
    final_reconstructed_paths = {}

    # Iterate over each connected component
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        
        if verbose:
            print(f"Processing connected component: {component}")
        
        # Simplify the subgraph
        G_simplified, full_path_corr = simplify_graph(subgraph)
        
        if verbose:
            print("Simplified graph edges:", G_simplified.edges())
        
        # Find all paths in the simplified subgraph
        all_paths_dict = find_all_paths(G_simplified)
        
        # Remove sub-paths from the list of all paths
        filtered_paths_dict = remove_subpaths(all_paths_dict)
        
        # Reconstruct the full paths from the filtered simplified paths
        reconstructed_paths_dict = reconstruct_full_paths(filtered_paths_dict, full_path_corr)
        
        # Add the reconstructed paths of the current component to the final result
        final_reconstructed_paths.update(reconstructed_paths_dict)
    
    if verbose:
        print("Final reconstructed paths:", final_reconstructed_paths)
    
    return final_reconstructed_paths

def reconstruct_paths(dimred_milestone2, verbose=False):
    """
    Reconstructs the full paths in a graph by simplifying it, finding all possible paths,
    removing redundant subpaths, and then expanding the paths based on intermediate nodes
    removed during simplification.

    This function performs the following steps:
    
    1. **Simplification of the Graph**: The input graph is simplified by collapsing nodes of degree 2,
       creating a correspondence of full paths that are hidden during the simplification process.
    2. **Finding All Paths in Simplified Graph**: All simple paths between any two nodes in the simplified
       graph are identified.
    3. **Filtering Paths to Remove Sub-Paths**: From the identified paths, sub-paths that are fully contained
       within other paths are removed, leaving only the longest unique paths.
    4. **Reconstruction of Full Paths**: Using the correspondence created during simplification, the full 
       paths are expanded to include all intermediate nodes removed during simplification.

    Parameters:
        dimred_milestone2 (pandas.DataFrame): A DataFrame containing the edges of a graph. It should have columns:
            - 'from': Source node of the edge
            - 'to': Target node of the edge
            - 'length': Weight or length of the edge
        verbose (bool): If True, enables verbose output for debugging and tracking the reconstruction process.

    Returns:
        dict: A dictionary where the keys are path names in the format "start-end" and the values are lists
              representing the fully reconstructed paths from the start node to the end node.
    """

    def simplify_graph(graph):
        """
        Simplifies the graph by iteratively removing nodes of degree 2 and storing their paths
        in the full_path_correspondence dictionary.

        Parameters:
            graph (networkx.Graph): The graph to be simplified.

        Returns:
            tuple:
                - networkx.Graph: The simplified graph.
                - dict: A dictionary storing full paths that correspond to the simplified edges.
        """
        # Dictionary to store the full path for each simplified edge
        full_path_correspondence = {}
        
        # Create a copy of the graph to modify
        G_simplified = graph.copy()
        
        # Continue simplifying while there are nodes with degree 2
        degree_2_nodes = [node for node in G_simplified.nodes() if G_simplified.degree(node) == 2]
        
        if verbose:
            print("Initial degree 2 nodes:", degree_2_nodes)
        
        while degree_2_nodes:
            # Take the first node with degree 2
            node = degree_2_nodes.pop()
            
            if verbose:
                print(f"Simplifying node: {node}")
            
            # Get its neighbors
            neighbors = list(G_simplified.neighbors(node))
            if len(neighbors) != 2:
                continue  # Safety check, should always be true for degree 2 nodes
            
            n1, n2 = neighbors
            
            # Create or extend the full path correspondence
            if (n1, n2) in full_path_correspondence:
                full_path_correspondence[(n1, n2)].insert(-1, node)
            elif (n2, n1) in full_path_correspondence:
                full_path_correspondence[(n2, n1)].insert(1, node)
            else:
                full_path_correspondence[(n1, n2)] = [n1, node, n2]
            
            # Remove the node from the graph
            G_simplified.remove_node(node)
            
            # Add an edge directly between the neighbors (n1 and n2)
            if not G_simplified.has_edge(n1, n2):
                G_simplified.add_edge(n1, n2)
            
            # Recalculate nodes with degree 2 in case simplification introduces new ones
            degree_2_nodes = [node for node in G_simplified.nodes() if G_simplified.degree(node) == 2]
            
            if verbose:
                print(f"Updated degree 2 nodes: {degree_2_nodes}")

        # Simplify the full path correspondence by merging nodes correctly
        for key in full_path_correspondence:
            unique_path = []
            for node in full_path_correspondence[key]:
                if node not in unique_path:
                    unique_path.append(node)
            full_path_correspondence[key] = unique_path
        
        if verbose:
            print("Final full path correspondence:", full_path_correspondence)

        return G_simplified, full_path_correspondence

    def find_all_paths(graph):
        """
        Finds all simple paths between any pair of nodes in the graph.

        Parameters:
            graph (networkx.Graph): The graph to find paths in.

        Returns:
            dict: A dictionary where the keys are in the format "start-end" and the values are lists
                  representing paths from the start node to the end node.
        """
        all_paths_dict = {}
        nodes = list(graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                source, target = nodes[i], nodes[j]
                all_paths = list(nx.all_simple_paths(graph, source, target))
                for path in all_paths:
                    key = f"{path[0]}-{path[-1]}"
                    all_paths_dict[key] = path
        
        if verbose:
            print("All paths found in the graph:", all_paths_dict)
        
        return all_paths_dict

    def remove_subpaths(all_paths_dict):
        """
        Filters out subpaths that are fully contained within other paths to keep only the longest unique paths.

        Parameters:
            all_paths_dict (dict): A dictionary containing all paths found in the graph.

        Returns:
            dict: A dictionary containing only the longest paths without subpaths.
        """
        all_paths = list(all_paths_dict.values())
        filtered_paths = set()
        for path in all_paths:
            if not any(set(path).issubset(set(other_path)) and path != other_path for other_path in all_paths):
                filtered_paths.add(tuple(path))
        
        filtered_paths_dict = {f"{path[0]}-{path[-1]}": list(path) for path in filtered_paths}
        
        if verbose:
            print("Filtered paths after removing sub-paths:", filtered_paths_dict)
        
        return filtered_paths_dict

    def expand_full_path(start_node, end_node, full_path_corr):
        """
        Recursively expands the full path between two nodes using the full_path_correspondence.

        Parameters:
            start_node (str): The starting node of the path segment.
            end_node (str): The ending node of the path segment.
            full_path_corr (dict): A dictionary containing the full paths of simplified edges.

        Returns:
            list: A list representing the fully expanded path segment.
        """
        if (start_node, end_node) in full_path_corr:
            segment = full_path_corr[(start_node, end_node)]
        elif (end_node, start_node) in full_path_corr:
            segment = full_path_corr[(end_node, start_node)][::-1]
        else:
            return [start_node, end_node]
        
        # Recursively expand each segment in the full path
        full_expanded_path = [segment[0]]
        for i in range(len(segment) - 1):
            start, end = segment[i], segment[i + 1]
            full_expanded_path.extend(expand_full_path(start, end, full_path_corr)[1:])
        
        return full_expanded_path

    def reconstruct_full_paths(filtered_paths_dict, full_path_corr):
        """
        Reconstructs the full paths by expanding all segments in the filtered paths dictionary.

        Parameters:
            filtered_paths_dict (dict): A dictionary containing the filtered paths from the simplified graph.
            full_path_corr (dict): A dictionary containing the full paths of simplified edges.

        Returns:
            dict: A dictionary where the keys are path names and the values are the fully reconstructed paths.
        """
        reconstructed_paths = {}
        for key, path in filtered_paths_dict.items():
            full_path = []
            for i in range(len(path) - 1):
                start_node = path[i]
                end_node = path[i + 1]
                
                # Expand the full path between consecutive nodes
                expanded_segment = expand_full_path(start_node, end_node, full_path_corr)
                
                # Add the expanded segment to the full path
                if full_path:
                    full_path.extend(expanded_segment[1:])
                else:
                    full_path.extend(expanded_segment)
            
            reconstructed_key = f"{full_path[0]}-{full_path[-1]}"
            reconstructed_paths[reconstructed_key] = full_path
        
        if verbose:
            print("Reconstructed full paths:", reconstructed_paths)
        
        return reconstructed_paths

    # Create the graph from the given edges
    G = nx.Graph()
    for _, row in dimred_milestone2.iterrows():
        G.add_edge(row['from'], row['to'], length=row['length'])
    
    if verbose:
        print("Initial graph created with edges:", G.edges(data=True))
    
    # Initialize the final reconstructed paths dictionary
    final_reconstructed_paths = {}

    # Iterate over each connected component
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        
        if verbose:
            print(f"Processing connected component: {component}")
        
        # Simplify the subgraph
        G_simplified, full_path_corr = simplify_graph(subgraph)
        
        if verbose:
            print("Simplified graph edges:", G_simplified.edges())
        
        # Find all paths in the simplified subgraph
        all_paths_dict = find_all_paths(G_simplified)
        
        # Remove sub-paths from the list of all paths
        filtered_paths_dict = remove_subpaths(all_paths_dict)
        
        # Reconstruct the full paths from the filtered simplified paths
        reconstructed_paths_dict = reconstruct_full_paths(filtered_paths_dict, full_path_corr)
        
        # Add the reconstructed paths of the current component to the final result
        final_reconstructed_paths.update(reconstructed_paths_dict)
    
    if verbose:
        print("Final reconstructed paths:", final_reconstructed_paths)
    
    # Check if there are values in final_reconstructed_paths
    if final_reconstructed_paths:
        # Return only the values of the final reconstructed paths
        return list(final_reconstructed_paths.values())
    else:
        # Return an empty list or any other preferred default value
        return []


def handle_cyclic_subgraph_old(subgraph, verbose=False):
    """
    Function to handle processing for cyclic subgraphs by finding and returning only the longest paths
    between any pair of nodes in the subgraph, removing any paths that are fully contained in longer paths.
    """
    all_paths = []
    nodes = list(subgraph.nodes)
    if verbose:
        print(f"handle_cyclic_subgraph")
        
    # Find all possible simple paths between each pair of nodes
    if verbose:
        print(f"Find all possible simple paths between each pair of nodes")

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            source = nodes[i]
            target = nodes[j]
            paths = list(nx.all_simple_paths(subgraph, source, target))
            all_paths.extend(paths)
    
    # Filter out subpaths that are fully contained within longer paths
    if verbose:
        print(f"Filter")
    longest_paths = []
    for path in all_paths:
        # If the path is not a subset of any longer path in the list, add it to longest_paths
        if not any(set(path).issubset(set(other_path)) and path != other_path for other_path in all_paths):
            longest_paths.append(path)
    
    if verbose:
        print(f"All longest paths in the cyclic subgraph: {longest_paths}")
    
    return longest_paths


def handle_cyclic_subgraph(subgraph, verbose=False):
    """
    Function to handle processing for cyclic subgraphs by finding and returning only the longest paths
    between any pair of nodes in the subgraph, removing any paths that are fully contained in longer paths.
    """
    if verbose:
        print("handle_cyclic_subgraph")

    all_paths = []
    nodes = list(subgraph.nodes)
    
    # Use a dictionary to cache paths found between pairs to avoid duplicate searches
    path_cache = {}

    # Find all simple paths between each pair of nodes
    if verbose:
        print("Find all possible simple paths between each pair of nodes")
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            source = nodes[i]
            target = nodes[j]
            
            # Check if this pair has already been processed
            if (source, target) in path_cache:
                continue
            
            # Get all simple paths from source to target
            paths = list(nx.all_simple_paths(subgraph, source, target))
            all_paths.extend(paths)
            
            # Cache the paths to avoid redundant calculations
            path_cache[(source, target)] = paths
            path_cache[(target, source)] = [list(reversed(p)) for p in paths]

    # Sort all paths by length in descending order for efficient filtering
    all_paths.sort(key=len, reverse=True)
    
    # Filter out subpaths that are fully contained within longer paths
    if verbose:
        print("Filter longest paths")

    longest_paths = []
    path_sets = [set(path) for path in all_paths]

    # Use a set to track added paths for faster lookup
    added_paths = set()
    
    for index, path in enumerate(all_paths):
        path_tuple = tuple(path)
        if path_tuple not in added_paths:
            longest_paths.append(path)
            # Mark all subpaths of the current path as added
            added_paths.update(tuple(other_path) for other_path_idx, other_path in enumerate(all_paths[index+1:], start=index+1)
                               if path_sets[other_path_idx].issubset(path_sets[index]))

    if verbose:
        print(f"All longest paths in the cyclic subgraph: {longest_paths}")
    
    return longest_paths


def extract_all_paths(dimred_milestone2, verbose=False):
    """
    Given a DataFrame `dimred_milestone2` with columns 'from', 'to', and 'length',
    this function extracts and returns all paths from both cyclic and non-cyclic subgraphs.

    Parameters:
        dimred_milestone2 (pd.DataFrame): DataFrame containing columns 'from', 'to', and 'length'.
        verbose (bool): If True, prints additional information about the number of cyclic and non-cyclic components.

    Returns:
        list: A list of paths from all subgraphs (both cyclic and non-cyclic).
    """
    # Create the full graph
    G = nx.Graph()
    for _, row in dimred_milestone2.iterrows():
        G.add_edge(row['from'], row['to'], length=row['length'])

    # Identify all connected components
    connected_components = list(nx.connected_components(G))

    # Initialize lists for paths and counts
    all_paths = []
    cyclic_components = []
    non_cyclic_components = []

    # Classify each component as cyclic or non-cyclic
    for component in connected_components:
        subgraph = G.subgraph(component)
        
        if not nx.cycle_basis(subgraph):  # Non-cyclic
            non_cyclic_components.append(subgraph)
        else:  # Cyclic
            cyclic_components.append(subgraph)

    # Print preliminary statistics if verbose is enabled
    if verbose:
        print(f"Number of cyclic components: {len(cyclic_components)}")
        print(f"Number of non-cyclic components: {len(non_cyclic_components)}")
        print(f"Total components: {len(cyclic_components) + len(non_cyclic_components)}")

    # Process non-cyclic components
    for subgraph in non_cyclic_components:
        # Create a DataFrame for the edges of the subgraph to pass to the function
        subgraph_edges = nx.to_pandas_edgelist(subgraph)
        
        # Rename columns to match the expected format
        subgraph_edges = subgraph_edges.rename(columns={'source': 'from', 'target': 'to'})
        
        # Call the function to handle non-cyclic subgraphs
        paths = reconstruct_paths(subgraph_edges, verbose=verbose)
        all_paths.extend(paths)

    # Process cyclic components
    N = 110
    N = 50
    for subgraph in cyclic_components:
        # Print message to indicate follow-up processing for cyclic graphs if verbose
        if verbose:
             print("Processing cyclic component...")

        # Check the number of nodes before handling the cyclic subgraph
        nnode_subgraph =  len(subgraph.nodes())
        if nnode_subgraph > N:
            if verbose:
                print(f"Skipping cyclic component has {nnode_subgraph} superior to {N} nodes limit.")
            continue

        # Call the function to handle cyclic subgraphs
        cyclic_paths = handle_cyclic_subgraph(subgraph, verbose=verbose)
        all_paths.extend(cyclic_paths)

    return all_paths
