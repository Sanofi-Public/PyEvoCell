import os
import numpy as np
import pandas as pd
from cellrank.kernels import PseudotimeKernel
from cellrank.estimators import GPCCA
import streamlit as st
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt

def cellrank_plots(adata_sub, g, celltype_column, folder):
    """
    Stores plots from cellrank

    """

    fig, ax = plt.subplots(figsize=(8, 6))

    # 2. Plot with Scanpy, telling it to use `ax` and not show immediately
    sc.pl.embedding(
        adata_sub,
        basis="tsne",
        color=[celltype_column, "dpt_pseudotime"],
        # ax=ax,
        show=False,  # Don't display it yet
    )

    # 3. Save to disk
    image_path1 = os.path.join(folder,"cellrank_pseudotimeplot.png")
    plt.savefig(image_path1, dpi=300)

    fig, ax = plt.subplots(figsize=(8, 6))
    g.plot_macrostates(which="terminal")
    image_path2 = os.path.join(folder, "fateprojections.png")
    plt.savefig(image_path2, dpi=300)

    # with plt.rc_context({"figure.figsize": (8, 6)}):
    #    sc.pl.embedding(adata_sub, basis="tsne", color=["CellCycle", "dpt_pseudotime"])
    # plt.gcf()
    # image_path1 = "www/images/cellrank_pseudotimeplot.png"
    # plt.savefig(image_path1)  # Save the figure as a PNG image

    # Perform rest of the computations
    g.compute_fate_probabilities()
    # g.plot_fate_probabilities(legend_loc="right")
    # cr.pl.circular_projection(adata_sub, keys="CellCycle", legend_loc="right")
    # plt.gcf()
    # image_path1 = "cellfates.png"
    # plt.savefig(image_path1)  # Save the figure as a PNG image

    # plt.gcf()
    # image_path2 = "fateprojections.png"
    # plt.savefig(image_path2)  # Save the figure as a PNG image

    print("Finished generating cellrank plots")
    return True

def compute_pseudotime(adata, starting_cellid):
    """
    Computes pseudotime and performs fltering for the cells in the path
    """

    try:
        # Compute Pseudotime
        sc.tl.diffmap(adata)
        # Select cell_id as the root to compute pseudotime
        adata.uns["iroot"] = adata.obs.index.get_loc(starting_cellid)
        # Calculate pseudotime
        sc.tl.dpt(adata)
        # Removing infinite values
        adata_sub = adata[np.isfinite(adata.obs["dpt_pseudotime"])]
        pseudotime_err_msg = ""
    except Exception as e:
        pseudotime_err_msg = e
        adata_sub = ""

    return adata_sub, pseudotime_err_msg

def create_cellrank_prompt(cellrank_prompt, drivers_df, terminal_celltype, exp_context):
    """
    Create prompt for the cellrank results
    """
    genes = drivers_df.index.to_list()
    genes = genes[0:25]
    genes = ", ".join(genes)
    print("*********************** In create_cellrank_prompt ************************")

    prompt = (
        f"The objective is to find the driver genes for the cell fates. The terminal cell state is listed within parenthesis ("
        + terminal_celltype
        + ")."
        + "The top 25 genes identified as drivers are listed within parenthesis ("
        + genes
        + "). "
        + " List genes that are responsible for cells to progress towards the terminal cell state "
        + terminal_celltype
        + "."
    )
    prompt = prompt + cellrank_prompt

    if exp_context is not None and len(exp_context) > 0:
        prompt = f"The context of the dataset is within parenthesis ({exp_context}). {prompt}"

    print(prompt)
    return prompt

def cellrank_computation(adata_sub, celltype_column):
    """
    Computes the transition matrix using a kernel, macrostates and terminal states
    """

    try:
        pk = PseudotimeKernel(adata_sub, time_key="dpt_pseudotime")
        pk.compute_transition_matrix()

        # Predict cell fates
        g = GPCCA(pk)
        # Construct macrostate
        print("Celltype Column: ", celltype_column)
        # Extract celltypes
        celltypes = list(adata_sub.obs[celltype_column].unique())
        g.fit(n_states=len(celltypes), cluster_key=celltype_column)
        print("macostates have been computed")
        print(adata_sub)
        # Predict Terminal states
        g.predict_terminal_states(method="top_n", n_states=6)
        print("g has been computed")
        err_msg = ""
    except Exception as e:
        # Print any errors that occur
        print(f"An error occurred in thread processing: {e}")
        err_msg = e
        g = ""

    return g, err_msg

def get_highest_percentage_cell(milestone_id, cell_type_value):
    """
    Returns the cell_id associated with the given milestone_id and CellCycle value,
    having the highest percentage.

    Args:
        milestone_id (str or int): The milestone ID to filter by.
        cell_type_value (str): The CellCycle value to filter by.

    Returns:
        str or None: The cell_id with the highest percentage, or None if no match is found.
    """
    # Ensure required data exists in session state
    if (
        "milestone_percentages" not in st.session_state
        or "cell_data" not in st.session_state
        or "selected_metadata" not in st.session_state
    ):
        return None

    # Merge milestone_percentages with cell_data on cell_id
    merged_df = st.session_state.milestone_percentages.merge(
        st.session_state.cell_data, on="cell_id", how="inner"
    )

    # Filter by milestone_id and cell_type_value
    filtered_df = merged_df[
        (merged_df["milestone_id"] == milestone_id)
        & (merged_df[st.session_state.selected_metadata] == cell_type_value)
    ]

    # Return the cell_id with the highest percentage, or None if no match found
    if not filtered_df.empty:
        return filtered_df.loc[filtered_df["percentage"].idxmax(), "cell_id"]

    return None

def get_cells_for_milestones(milestone_list):
    """
    Returns a list of cell_ids associated with the given list of milestone_ids and CellCycle value.

    Args:
        milestone_list (list): A list of milestone IDs to filter by.
        cell_type_value (str): The CellCycle value to filter by.

    Returns:
        list: A list of cell_id values matching the criteria.
    """
    # Ensure required data exists in session state
    if (
        "milestone_percentages" not in st.session_state
        or "cell_data" not in st.session_state
        or "selected_metadata" not in st.session_state
    ):
        return None  # pylint: disable=R1705

    # Merge milestone_percentages with cell_data on cell_id
    merged_df = st.session_state.milestone_percentages.merge(
        st.session_state.cell_data, on="cell_id", how="inner"
    )

    # Filter by milestone_list and cell_type_value
    filtered_df = merged_df[(merged_df["milestone_id"].isin(milestone_list))]

    # Return the list of unique cell_ids
    return filtered_df["cell_id"].unique().tolist()

def normalize_data(cell_metadata, counts_matrix, celltype_column, cellids_in_path):
    """
    Constructs an AnnData Object and noemalizes the data
    """
    # Making the celltype column categorical
    cell_metadata[celltype_column] = pd.Categorical(cell_metadata[celltype_column])

    # Create AnnData Object
    adata = ad.AnnData(X=counts_matrix.T, obs=cell_metadata.set_index("cell_id"))
    print(celltype_column)

    # Use cellids that are in the selected path
    adata = adata[adata.obs_names.isin(cellids_in_path)]

    # Normalize
    sc.pp.filter_genes(adata, min_cells=15)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Feature selection
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    # Dimensionality Reduction
    sc.tl.pca(adata, random_state=0)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    # Tsne
    sc.tl.tsne(adata, n_pcs=30)

    return adata