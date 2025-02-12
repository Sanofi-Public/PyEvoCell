import pandas as pd
import os

def load_prompts(script_dir):
    prompts_df_path = os.path.join(script_dir, "llm/prompts.csv")
    prompts_df = pd.read_csv(prompts_df_path)
    prompts = dict(zip(prompts_df.iloc[:, 0], prompts_df.iloc[:, 1]))
    return prompts

# Function to load data from CSV files
def load_cds_from_csv(directory):
    # Read the required CSV files from the selected directory
    trajectory_edges = pd.read_csv(os.path.join(directory, "trajectory_edges.csv"))

    cell_metadata = pd.read_csv(os.path.join(directory, "cell_metadata.csv"), dtype=str)
    dimred = pd.read_csv(os.path.join(directory, "dimred.csv"))
    vertex_coordinates = pd.read_csv(os.path.join(directory, "dimred_milestone.csv"))
    milestone_percentages = pd.read_csv(
        os.path.join(directory, "milestone_percentages.csv")
    )

    # Merge dataframes on 'cell_id'
    cell_data = pd.merge(cell_metadata, dimred, on="cell_id")

    # Create the dimensional reduction matrix for cells
    dimred = cell_data[["comp_1", "comp_2"]]
    dimred.index = cell_data["cell_id"]

    # Create the dimensional reduction matrix for vertices (milestones)
    dimred_milestones = vertex_coordinates[["comp_1", "comp_2"]]
    dimred_milestones.index = vertex_coordinates["milestone_id"]

    # Construct milestone network (graph edges) using the trajectory edges
    milestone_network2 = trajectory_edges

    # Read Count Data from CSV
    counts_matrix = pd.read_csv(os.path.join(directory, "count_data.csv"), index_col=0)

    return (
        cell_data,
        milestone_network2,
        dimred,
        dimred_milestones,
        milestone_percentages,
        counts_matrix,
    )