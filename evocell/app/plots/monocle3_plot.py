import os
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# Function to create a plotly graph with colored cells based on metadata
def create_plot(milestone_network, dimred, dimred_milestones, cell_data, color_by=None):
    # Create figure
    fig = make_subplots()

    # If a metadata column is chosen, use it to color the cells
    if color_by:
        # Get unique values for the metadata
        if pd.api.types.is_numeric_dtype(cell_data[color_by]):
            # Continuous metadata: use color scale
            cell_colors = cell_data[color_by]
            fig.add_trace(
                go.Scatter(
                    x=dimred["comp_1"],
                    y=dimred["comp_2"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=cell_colors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=color_by),
                    ),
                    name="Cells",
                    text=cell_data["cell_id"],
                )
            )
        else:
            # Discrete metadata: plot each category separately for the legend
            unique_values = cell_data[color_by].unique()
            # Generate a color palette for the unique values
            color_palette = px.colors.qualitative.Plotly
            color_map = {
                val: color_palette[i % len(color_palette)]
                for i, val in enumerate(unique_values)
            }

            # Plot cells by category to display them in the legend
            for value in unique_values:
                subset = cell_data[cell_data[color_by] == value]
                fig.add_trace(
                    go.Scatter(
                        x=dimred.loc[subset["cell_id"], "comp_1"],
                        y=dimred.loc[subset["cell_id"], "comp_2"],
                        mode="markers",
                        marker=dict(size=5, color=color_map[value]),
                        name=value,
                        text=subset["cell_id"],
                    )
                )
    else:
        # Default to a single color
        fig.add_trace(
            go.Scatter(
                x=dimred["comp_1"],
                y=dimred["comp_2"],
                mode="markers",
                marker=dict(size=5, color="blue"),
                name="Cells",
                text=cell_data["cell_id"],
            )
        )

    # Plot milestone points (dimred_milestones)
    fig.add_trace(
        go.Scatter(
            x=dimred_milestones["comp_1"],
            y=dimred_milestones["comp_2"],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Milestones",
        )
    )

    # Plot milestone network edges
    for _, row in milestone_network.iterrows():
        source = row["from"]
        target = row["to"]

        # Get coordinates for from and target
        x0, y0 = dimred_milestones.loc[source, ["comp_1", "comp_2"]]
        x1, y1 = dimred_milestones.loc[target, ["comp_1", "comp_2"]]

        # Add line segment between milestones
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,  # Do not add edges to the legend
            )
        )

    # Update layout
    fig.update_layout(
        title="Cells and Milestone Network",
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        width=800,
        height=600,
    )

    return fig


# Function to create a plotly graph with colored cells based on metadata and all paths
def create_plot_with_paths_old(
    milestone_network, dimred, dimred_milestones, cell_data, all_paths, color_by=None
):
    # Create figure
    fig = make_subplots()

    # If a metadata column is chosen, use it to color the cells
    if color_by:
        # Get unique values for the metadata
        if pd.api.types.is_numeric_dtype(cell_data[color_by]):
            # Continuous metadata: use color scale
            cell_colors = cell_data[color_by]
            fig.add_trace(
                go.Scatter(
                    x=dimred["comp_1"],
                    y=dimred["comp_2"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=cell_colors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=color_by),
                    ),
                    name="Cells",
                    text=cell_data["cell_id"],
                )
            )
        else:
            # Discrete metadata: plot each category separately for the legend
            unique_values = cell_data[color_by].unique()
            # Generate a color palette for the unique values
            color_palette = px.colors.qualitative.Plotly
            color_map = {
                val: color_palette[i % len(color_palette)]
                for i, val in enumerate(unique_values)
            }

            # Plot cells by category to display them in the legend
            for value in unique_values:
                subset = cell_data[cell_data[color_by] == value]
                fig.add_trace(
                    go.Scatter(
                        x=dimred.loc[subset["cell_id"], "comp_1"],
                        y=dimred.loc[subset["cell_id"], "comp_2"],
                        mode="markers",
                        marker=dict(size=5, color=color_map[value]),
                        name=value,
                        text=subset["cell_id"],
                    )
                )
    else:
        # Default to a single color
        fig.add_trace(
            go.Scatter(
                x=dimred["comp_1"],
                y=dimred["comp_2"],
                mode="markers",
                marker=dict(size=5, color="blue"),
                name="Cells",
                text=cell_data["cell_id"],
            )
        )

    # Plot milestone points (dimred_milestones)
    fig.add_trace(
        go.Scatter(
            x=dimred_milestones["comp_1"],
            y=dimred_milestones["comp_2"],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Milestones",
        )
    )

    # Plot milestone network edges
    for _, row in milestone_network.iterrows():
        source = row["from"]
        target = row["to"]

        # Get coordinates for from and target
        x0, y0 = dimred_milestones.loc[source, ["comp_1", "comp_2"]]
        x1, y1 = dimred_milestones.loc[target, ["comp_1", "comp_2"]]

        # Add line segment between milestones
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,  # Do not add edges to the legend
            )
        )

    # Plot paths from all_paths
    path_colors = px.colors.qualitative.Dark24  # Choose a color palette for paths
    for i, (path_key, path_nodes) in enumerate(all_paths.items()):
        # Extract coordinates for each path segment
        path_color = path_colors[i % len(path_colors)]
        for j in range(len(path_nodes) - 1):
            source = path_nodes[j]
            target = path_nodes[j + 1]
            x0, y0 = dimred_milestones.loc[source, ["comp_1", "comp_2"]]
            x1, y1 = dimred_milestones.loc[target, ["comp_1", "comp_2"]]

            # Add line segments for each path
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=path_color, width=2, dash="dash"),
                    name=f"Path {path_key}",
                    showlegend=(
                        j == 0
                    ),  # Only show legend entry for the first segment of each path
                )
            )

    # Update layout
    fig.update_layout(
        title="Cells, Milestone Network, and Paths",
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        width=800,
        height=600,
    )

    return fig


def create_plot_with_paths_old2(
    milestone_network, dimred, dimred_milestones, cell_data, all_paths, color_by=None
):
    # Create figure
    fig = make_subplots()

    # If a metadata column is chosen, use it to color the cells
    if color_by:
        # Get unique values for the metadata
        if pd.api.types.is_numeric_dtype(cell_data[color_by]):
            # Continuous metadata: use color scale
            cell_colors = cell_data[color_by]
            fig.add_trace(
                go.Scatter(
                    x=dimred["comp_1"],
                    y=dimred["comp_2"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=cell_colors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=color_by),
                    ),
                    name="Cells",
                    text=cell_data["cell_id"],
                )
            )
        else:
            # Discrete metadata: plot each category separately for the legend
            unique_values = cell_data[color_by].unique()
            # Generate a color palette for the unique values
            color_palette = px.colors.qualitative.Plotly
            color_map = {
                val: color_palette[i % len(color_palette)]
                for i, val in enumerate(unique_values)
            }

            # Plot cells by category to display them in the legend
            for value in unique_values:
                subset = cell_data[cell_data[color_by] == value]
                fig.add_trace(
                    go.Scatter(
                        x=dimred.loc[subset["cell_id"], "comp_1"],
                        y=dimred.loc[subset["cell_id"], "comp_2"],
                        mode="markers",
                        marker=dict(size=5, color=color_map[value]),
                        name=value,
                        text=subset["cell_id"],
                    )
                )
    else:
        # Default to a single color
        fig.add_trace(
            go.Scatter(
                x=dimred["comp_1"],
                y=dimred["comp_2"],
                mode="markers",
                marker=dict(size=5, color="blue"),
                name="Cells",
                text=cell_data["cell_id"],
            )
        )

    # Plot milestone points (dimred_milestones)
    fig.add_trace(
        go.Scatter(
            x=dimred_milestones["comp_1"],
            y=dimred_milestones["comp_2"],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Milestones",
        )
    )

    # Plot milestone network edges
    for _, row in milestone_network.iterrows():
        source = row["from"]
        target = row["to"]

        # Get coordinates for from and target
        x0, y0 = dimred_milestones.loc[source, ["comp_1", "comp_2"]]
        x1, y1 = dimred_milestones.loc[target, ["comp_1", "comp_2"]]

        # Add line segment between milestones
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,  # Do not add edges to the legend
            )
        )

    # Plot paths from all_paths using nodes
    path_nodes = all_paths["node"]
    path_colors = px.colors.qualitative.Dark24  # Choose a color palette for paths
    path_color = path_colors[
        0
    ]  # Using one color for simplicity; can change if multiple paths exist

    # Iterate through the node pairs to plot the path
    for i in range(len(path_nodes) - 1):
        source = path_nodes[i]
        target = path_nodes[i + 1]
        x0, y0 = dimred_milestones.loc[source, ["comp_1", "comp_2"]]
        x1, y1 = dimred_milestones.loc[target, ["comp_1", "comp_2"]]

        # Add line segments for the path
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=path_color, width=2, dash="dash"),
                name=f"Path",
                showlegend=(
                    i == 0
                ),  # Only show legend entry for the first segment of the path
            )
        )

    # Update layout
    fig.update_layout(
        title="Cells, Milestone Network, and Path",
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        width=800,
        height=600,
    )

    return fig


def create_plot_with_paths(
    milestone_network, dimred, dimred_milestones, cell_data, all_paths, color_by=None
):
    # Create figure
    fig = make_subplots()

    # If a metadata column is chosen, use it to color the cells
    if color_by:
        # Get unique values for the metadata
        if pd.api.types.is_numeric_dtype(cell_data[color_by]):
            # Continuous metadata: use color scale
            cell_colors = cell_data[color_by]
            fig.add_trace(
                go.Scatter(
                    x=dimred["comp_1"],
                    y=dimred["comp_2"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=cell_colors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=color_by),
                    ),
                    name="Cells",
                    text=cell_data["cell_id"],
                )
            )
        else:
            # Discrete metadata: plot each category separately for the legend
            unique_values = cell_data[color_by].unique()
            # Generate a color palette for the unique values
            color_palette = px.colors.qualitative.Plotly
            color_map = {
                val: color_palette[i % len(color_palette)]
                for i, val in enumerate(unique_values)
            }

            # Plot cells by category to display them in the legend
            for value in unique_values:
                subset = cell_data[cell_data[color_by] == value]
                fig.add_trace(
                    go.Scatter(
                        x=dimred.loc[subset["cell_id"], "comp_1"],
                        y=dimred.loc[subset["cell_id"], "comp_2"],
                        mode="markers",
                        marker=dict(size=5, color=color_map[value]),
                        name=value,
                        text=subset["cell_id"],
                    )
                )
    else:
        # Default to a single color
        fig.add_trace(
            go.Scatter(
                x=dimred["comp_1"],
                y=dimred["comp_2"],
                mode="markers",
                marker=dict(size=5, color="blue"),
                name="Cells",
                text=cell_data["cell_id"],
            )
        )

    # Plot milestone points (dimred_milestones) in blue for clear distinction
    fig.add_trace(
        go.Scatter(
            x=dimred_milestones["comp_1"],
            y=dimred_milestones["comp_2"],
            mode="markers",
            marker=dict(size=10, color="blue"),
            name="Milestones",
        )
    )

    # Plot milestone network edges
    for _, row in milestone_network.iterrows():
        source = row["from"]
        target = row["to"]

        # Get coordinates for from and target
        x0, y0 = dimred_milestones.loc[source, ["comp_1", "comp_2"]]
        x1, y1 = dimred_milestones.loc[target, ["comp_1", "comp_2"]]

        # Add line segment between milestones
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="grey", width=2),
                showlegend=False,  # Do not add edges to the legend
            )
        )

    # Plot paths from all_paths using nodes in black
    path_nodes = all_paths["node"]
    path_color = "black"  # Use black for the path

    # Iterate through the node pairs to plot the path
    for i in range(len(path_nodes) - 1):
        source = path_nodes[i]
        target = path_nodes[i + 1]
        x0, y0 = dimred_milestones.loc[source, ["comp_1", "comp_2"]]
        x1, y1 = dimred_milestones.loc[target, ["comp_1", "comp_2"]]

        # Add line segments for the path in black
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines+markers",
                line=dict(color=path_color, width=2, dash="dash"),
                marker=dict(
                    size=8, color=path_color, symbol="circle"
                ),  # Black markers for path nodes
                name=f"Path",
                showlegend=(
                    i == 0
                ),  # Only show legend entry for the first segment of the path
            )
        )

    # Update layout
    fig.update_layout(
        title="Cells, Milestone Network, and Path",
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        width=800,
        height=600,
    )

    return fig


# Function to create a plotly graph with colored cells based on metadata without milestones
def create_plot_nomilestones(
    milestone_network, dimred, dimred_milestones, cell_data, color_by=None
):
    # Create figure
    fig = make_subplots()

    # If a metadata column is chosen, use it to color the cells
    if color_by:
        # Get unique values for the metadata
        if pd.api.types.is_numeric_dtype(cell_data[color_by]):
            # Continuous metadata: use color scale
            cell_colors = cell_data[color_by]
            fig.add_trace(
                go.Scatter(
                    x=dimred["comp_1"],
                    y=dimred["comp_2"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=cell_colors,
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title=color_by),
                    ),
                    name="Cells",
                    text=cell_data["cell_id"],
                )
            )
        else:
            # Discrete metadata: plot each category separately for the legend
            unique_values = cell_data[color_by].unique()
            # Generate a color palette for the unique values
            color_palette = px.colors.qualitative.Plotly
            color_map = {
                val: color_palette[i % len(color_palette)]
                for i, val in enumerate(unique_values)
            }

            # Plot cells by category to display them in the legend
            for value in unique_values:
                subset = cell_data[cell_data[color_by] == value]
                fig.add_trace(
                    go.Scatter(
                        x=dimred.loc[subset["cell_id"], "comp_1"],
                        y=dimred.loc[subset["cell_id"], "comp_2"],
                        mode="markers",
                        marker=dict(size=5, color=color_map[value]),
                        name=value,
                        text=subset["cell_id"],
                    )
                )
    else:
        # Default to a single color
        fig.add_trace(
            go.Scatter(
                x=dimred["comp_1"],
                y=dimred["comp_2"],
                mode="markers",
                marker=dict(size=5, color="blue"),
                name="Cells",
                text=cell_data["cell_id"],
            )
        )

    # Plot milestone points (dimred_milestones)
    # fig.add_trace(
    #    go.Scatter(
    #        x=dimred_milestones["comp_1"],
    #        y=dimred_milestones["comp_2"],
    #        mode="markers",
    #        marker=dict(size=10, color="red"),
    #        name="Milestones",
    #    )
    # )

    # Plot milestone network edges
    for _, row in milestone_network.iterrows():
        source = row["from"]
        target = row["to"]

        # Get coordinates for from and target
        x0, y0 = dimred_milestones.loc[source, ["comp_1", "comp_2"]]
        x1, y1 = dimred_milestones.loc[target, ["comp_1", "comp_2"]]

        # Add line segment between milestones
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,  # Do not add edges to the legend
            )
        )

    # fig.update_traces(
    #    hovertemplate="%{customdata[0]}", customdata=cell_data[[color_by]]
    # )

    # Update layout
    fig.update_layout(
        title="Monocle Trajectory",
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        width=800,
        height=600,
    )

    return fig
