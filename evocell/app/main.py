import os
import pandas as pd
import streamlit as st
from PIL import Image
import base64
import threading
import time
import re
import llm.process_hypothesis
from llm.provider import get_llm_output, get_available_models, execute_ollama_script
from search_path import search_path
from plots.monocle3_plot import create_plot, create_plot_with_paths, create_plot_nomilestones
from st_aggrid import GridOptionsBuilder, AgGrid
import anndata as ad
import scanpy as sc
import cellrank as cr
import numpy as np
import matplotlib.pyplot as plt
from utils import (
    file_utils, st_utils, explainpath_utils, dge_utils,
    drivergenes_utils, gsea_utils, veracity_utils)
import warnings
import queue
import gseapy as gp
import urllib.parse

# Initialize variables, models and prompts
st_utils.initialize_session()
llm_models_list = get_available_models()
progress_queue = queue.Queue()
script_dir = os.path.dirname(os.path.abspath(__file__))
prompts = file_utils.load_prompts(script_dir)

# Streamlit app
def main():

    # Load and resize the logo
    logo_path = os.path.join(script_dir, "../www/images/mdm_product_logo.png")
    logo = Image.open(logo_path)

    # Create columns for logo and title
    col1, col2 = st.columns([5, 40])  # Adjust width ratios as needed

    # Display logo in the first column
    with col1:
        st.image(logo, use_column_width=True)

    # Display title in the second column
    with col2:
        st.title("EvoCell")

    # Create tabs
    tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Load Dataset",
            "Generate Hypothesis",
            "Search Path",
            "Explain Path",
            "DGE",
            "Driver Genes",
            "GSEA",
            "Veracity Filter",
        ]
    )

    # Content for Tab 0
    with tab0:

        session = st.runtime.get_instance()._session_mgr.list_active_sessions()[0]
        print(session.client.request)
        st_base_url = urllib.parse.urlunparse(
            [
                session.client.request.protocol,
                session.client.request.host,
                "",
                "",
                "",
                "",
            ]
        )

        print(st_base_url)

        # Select directory
        input_dir = st.text_input("Enter the directory containing CSV files:", "")
       
        if input_dir:
            if os.path.exists(input_dir):
                # Run when input_dir has changed
                if st.session_state.previous_input_dir != input_dir:
                    with st.spinner("Loading dataset from csv... PLease wait."):
                        print("Loading dataset from csv")
                        # Reset session state variables if input_dir is different
                        st.session_state.previous_input_dir = input_dir
                        st.session_state.all_paths = None

                        # Load data from CSV files
                        (
                            cell_data,
                            milestone_network2,
                            dimred,
                            dimred_milestones,
                            milestone_percentages,
                            counts_matrix,
                        ) = file_utils.load_cds_from_csv(input_dir)

                        # Store headers and processed_rows in session state
                        st.session_state.cell_data = cell_data
                        st.session_state.milestone_network2 = milestone_network2
                        st.session_state.dimred = dimred
                        st.session_state.dimred_milestones = dimred_milestones
                        st.session_state.milestone_percentages = milestone_percentages
                        st.session_state.counts_matrix = counts_matrix

                        # Extract paths
                        st.session_state.all_paths = search_path.extract_all_paths(
                            milestone_network2, verbose=True
                        )

                 # Text area for entering a few words on two lines
                context_input = st.text_area(
                    "Enter the context of your study here:", ""  # , height=50
                )
                st.session_state.context_input = context_input

                selected_llm_model = st.selectbox("Select llm model:", llm_models_list, index=0)
                st.session_state.selected_llm_models = selected_llm_model

                # Display some basic info about the loaded data
                st.markdown("#### Overview of Cell Data")
                cell_data = st.session_state.cell_data

                # Display shape of the data
                st.write(f"Number of cells: {cell_data.shape[0]}")
                st.write(f"Number of metadata: {cell_data.shape[1]}")

                # Display a sample of the data
                st.write("**Sample Data:**")
                st.dataframe(cell_data.head())

                cell_data = st.session_state.cell_data

                # Select metadata column for further processing
                metadata_columns = cell_data.columns.difference(
                    ["comp_1", "comp_2", "cell_id"]
                )

                # Time column
                time_column = st.selectbox(
                    "Select time column (optional):",
                    [""] + list(metadata_columns),
                    index=0,
                )
                st.session_state.time_column = time_column

                # Celltype column
                selected_metadata = st.selectbox(
                    "Select celltype metadata for analysis:",
                    [""] + list(metadata_columns),
                    index=0,
                )

                # Check if a valid column has been selected
                if selected_metadata != "":
                    st.session_state.selected_metadata = selected_metadata

                    # Get unique values for the selected metadata column
                    unique_metadata_values = cell_data[selected_metadata].unique()

                    # Create and display a button to show the Plotly graph
                    if st.button("Display Trajectory"):
                        # Create plotly graph
                        fig = create_plot(
                            st.session_state.milestone_network2,
                            st.session_state.dimred,
                            st.session_state.dimred_milestones,
                            st.session_state.cell_data,
                            color_by=selected_metadata,
                        )
                        st.plotly_chart(fig)

            else:
                # Show an error message if directory is invalid
                st.error("Invalid directory. Please check the path and try again.")

    # Content for Tab 1
    with tab1:
        # Check if cell_data exists in session state
        if "cell_data" in st.session_state and st.session_state.cell_data is not None:
            # Use the loaded cell_data from session state
            cell_data = st.session_state.cell_data

            # Generate a comma-separated string of unique values in the selected metadata column
            if selected_metadata:
                unique_metadata_values = cell_data[selected_metadata].unique()
                unique_metadata_values = list(map(str, unique_metadata_values))
                unique_value_count = len(unique_metadata_values)

                # Restrict selection if there are more than 30 unique values
                if unique_value_count > 30:
                    st.warning(
                        f"The metadata '{selected_metadata}' contains {unique_value_count} unique values, which is too many to analyze. Please select another column."
                    )
                else:
                    unique_metadata_string = ", ".join(unique_metadata_values)
                    unique_metadata_values = [
                        value.replace(";", "") for value in unique_metadata_values
                    ]
                    unique_metadata_values = [
                        value.replace(":", "") for value in unique_metadata_values
                    ]

                    # Display the cell states
                    st.write("### Potential cell state transitions in your dataset")
                    st.write(
                        "Cell types in your dataset include: " + unique_metadata_string
                    )

                    # Add a button to trigger the execution
                    if st.button("Generate Hypothesis"):

                        prompt = prompts.get("Hypothesis generation")
                        prompt = prompt + unique_metadata_string
                        if (
                            st.session_state.context_input is not None
                            and len(st.session_state.context_input) > 0
                        ):
                            prompt = f"The context of the dataset is within parenthesis ({st.session_state.context_input}). {prompt}"

                        print(prompt)
                        # Execute the LLM script to get the answer differentiation table
                        with st.spinner("Generating hypothesis with LLM..."):
                            answer_differentiation_table = get_llm_output(
                                llm_model=st.session_state.selected_llm_models,
                                input_string=prompt,
                            )

                        # Initialize session state for progress tracking
                        st.session_state["processing_progress"] = ""

                        # Local storage for thread results
                        results = {}

                        # Launch a separate thread to process hypothesis data
                        processing_thread = threading.Thread(
                            target=llm.process_hypothesis.process_hypothesis_data_in_thread,
                            args=(
                                st.session_state.selected_llm_models,
                                answer_differentiation_table,
                                results,
                                progress_queue
                            ),
                            daemon=True,
                        )
                        processing_thread.start()

                        # Initialize a variable to track old progress
                        old_progress = None

                        progress_placeholder = st.empty()

                        with st.spinner("Processing hypothesis data..."):
                            while (
                                processing_thread.is_alive()
                                or not progress_queue.empty()
                            ):
                                try:
                                    # Check for any messages in the queue
                                    current_progress = progress_queue.get_nowait()

                                    # Only update the placeholder if there's a change
                                    if current_progress != old_progress:
                                        progress_placeholder.text(current_progress)
                                        old_progress = current_progress
                                except queue.Empty:
                                    # No message, continue
                                    pass

                                time.sleep(0.5)

                            # Once the thread is done, update final message
                            progress_placeholder.text("")

                        # After processing is complete, handle transferring results to session state
                        if "headers" in results and "processed_rows" in results:
                            st.session_state.headers = results["headers"]
                            st.session_state.processed_rows = results["processed_rows"]
                        else:
                            st.info(
                                "There was an error getting the LLM response. Please try generating hypothesis again."
                            )

                        if st.session_state.headers and st.session_state.processed_rows:
                            # Create two DataFrames: one for all rows and one for valid rows
                            # valid rows (publication found): st.session_state.generated_hypothesis_df
                            # all rows: st.session_state.df
                            all_data = [
                                (cell_state, target_state, publication_found, citation)
                                for cell_state, target_state, publication_found, citation in st.session_state.processed_rows
                            ]
                            valid_rows = [
                                row for row in st.session_state.processed_rows if row[2]
                            ]  # Assuming 'is_valid' is the third item in the tuple

                            # Data for st.session_state.df (all rows)
                            # If 'st.session_state.headers' does not include 'citation', add it
                            if len(st.session_state.headers) == 3:
                                headers_with_citation = st.session_state.headers + [
                                    "citation"
                                ]
                            else:
                                headers_with_citation = st.session_state.headers

                            all_data_df = pd.DataFrame(
                                all_data, columns=headers_with_citation
                            )  # Include 'citation' column

                            # Data for st.session_state.generated_hypothesis_df (only valid rows, excluding 'citation')
                            valid_data = [
                                (cell_state, target_state, publication_found)
                                for cell_state, target_state, publication_found, citation in valid_rows
                            ]
                            new_df = pd.DataFrame(
                                valid_data, columns=st.session_state.headers[:3]
                            )  # Exclude 'citation' column

                            # Store the filtered DataFrame (without citation) in session state for valid rows only
                            st.session_state.generated_hypothesis_df = new_df

                            # Save index and citation in session state
                            st.session_state.citations = {
                                index: citation
                                for index, (
                                    cell_state,
                                    target_state,
                                    publication_found,
                                    citation,
                                ) in enumerate(valid_rows)
                                if citation
                            }

                            # Display the generated_hypothesis_df DataFrame
                            # st.dataframe(new_df)

                            # Display citations below the DataFrame for valid rows
                            # st.write("Citations:")
                            # for index, citation in st.session_state.citations.items():
                            #    st.write(f"Row {index}: {citation}")

                            # The following block keeps all rows in st.session_state.df
                            all_data_df["Selected"] = False
                            st.session_state.df = all_data_df  # if the user wants to keep previous hypothesis, comment this, uncomment following block

                            # # If st.session_state.df already exists and is not empty
                            # if (
                            #     "df" in st.session_state
                            #     and not st.session_state.df.empty
                            # ):
                            #     # Combine the all_data_df with the existing st.session_state.df
                            #     combined_df = pd.concat(
                            #         [st.session_state.df, all_data_df],
                            #         ignore_index=True,
                            #     )
                            #     # Drop duplicates to keep only unique rows
                            #     st.session_state.df = combined_df.drop_duplicates()
                            # else:
                            #     # If st.session_state.df is empty, simply set it as all_data_df
                            #     st.session_state.df = all_data_df

                            # Display the st.session_state.df DataFrame with all rows
                            # st.dataframe(st.session_state.df)

                        else:
                            # If headers and processed_rows are not available, initialize an empty DataFrame
                            st.session_state.df = pd.DataFrame(
                                columns=[
                                    "Cell state",
                                    "Can Differentiate Into",
                                    "Publication Found",
                                    "Selected",
                                ]
                            )
                            st.session_state.generated_hypothesis_df = pd.DataFrame(
                                columns=[
                                    "Cell state",
                                    "Can Differentiate Into",
                                    "Publication Found",
                                ]
                            )
                            st.write(
                                "None of the cell state transitions considered were plausible."
                            )

                    # Display results when available
                    if (
                        "generated_hypothesis_df" in st.session_state
                        and "df" in st.session_state
                        and len(st.session_state.df) > 0
                    ):
                        st.markdown("#### Validated hypothesis")
                        if len(st.session_state.generated_hypothesis_df) > 0:
                            st.dataframe(st.session_state.generated_hypothesis_df)
                            # Display citations below the DataFrame for valid rows
                            st.write("Citations:")
                            for index, citation in st.session_state.citations.items():
                                st.write(f"Row {index}: {citation}")
                        else:
                            st.write(
                                "There were no cell state transitions backed by PubMed publications."
                            )
                        st.markdown("#### List of possible hypothesis")
                        st.dataframe(
                            st.session_state.df.drop(
                                columns="Selected", errors="ignore"
                            )
                        )

                    # Display the previously generated DataFrame if it exists
                    # elif "generated_hypothesis_df" in st.session_state:
                    #    st.dataframe(st.session_state.generated_hypothesis_df)
            #
            #    # Display citations below the DataFrame for previously generated data
            #    if "citations" in st.session_state:
            #        st.write("Citations:")
            #        for index, citation in st.session_state.citations.items():
            #            st.write(f"Row {index}: {citation}")

        else:
            # Your alternative logic, if needed
            st.warning("Please select the celltype metada column in Load Dataset tab.")

    # Content for Tab 2
    with tab2:
        # Check if cell_data exists in session state
        if (
            "cell_data" in st.session_state
            and st.session_state.cell_data is not None
            and "generated_hypothesis_df" in st.session_state
        ):
            # Use the loaded cell_data from session state
            cell_data = st.session_state.cell_data

            # Ensure headers and processed_rows are in session state
            # Check if generated_hypothesis_df exists in session state
            if "generated_hypothesis_df" in st.session_state:
                # Check if df has more rows than generated_hypothesis_df; if not, proceed
                if "df" not in st.session_state or len(st.session_state.df) <= len(
                    st.session_state.generated_hypothesis_df
                ):
                    # Duplicate generated_hypothesis_df to df
                    st.session_state.df = (
                        st.session_state.generated_hypothesis_df.copy()
                    )

                    # Add a new column 'selected' initialized to False
                    st.session_state.df["Selected"] = False
            else:
                # If generated_hypothesis_df does not exist, initialize an empty DataFrame with 'selected' column
                st.session_state.df = pd.DataFrame(
                    columns=[
                        "Cell state",
                        "Can Differentiate Into",
                        "Publication Found",
                        "selected",
                    ]
                )

            # Display DataFrame in Streamlit
            st.write("### Transitions from Generate Hypothesis")
            st.dataframe(st.session_state.generated_hypothesis_df)

            # Ensure the DataFrame is not empty and contains the correct columns before proceeding
            if (
                not st.session_state.generated_hypothesis_df.empty
                and "Cell state" in st.session_state.generated_hypothesis_df.columns
                and "Can Differentiate Into"
                in st.session_state.generated_hypothesis_df.columns
            ):
                # List of unique cell types for dropdowns
                cell_types = list(
                    set(st.session_state.generated_hypothesis_df["Cell state"]).union(
                        st.session_state.generated_hypothesis_df[
                            "Can Differentiate Into"
                        ]
                    )
                )
            else:
                st.warning(
                    "The DataFrame is empty or does not contain the expected columns."
                )
                cell_types = []

            cell_types_to_add = []

            # Check if the selected metadata exists in session state and is not None
            if st.session_state.get("selected_metadata"):
                cell_types_to_add = sorted(
                    cell_data[st.session_state.selected_metadata].unique()
                )

            # Editable table for users to add rows
            st.subheader("Add Cell Transition")
            with st.form(key="add_row_form"):
                new_cell_state = st.selectbox("Cell state", cell_types_to_add, index=0)
                new_transition_to = st.selectbox(
                    "Can Differentiate Into", cell_types_to_add, index=0
                )
                submit_button = st.form_submit_button(label="Add Cell Path")

            # Add new row to DataFrame if form is submitted
            if submit_button:
                print("\nAdding new row\n")
                # Check if the two selected values are the same
                if new_cell_state == new_transition_to:
                    st.error(
                        "The 'Cell state' and 'Can Differentiate Into' must be different."
                    )
                else:
                    new_row = pd.DataFrame(
                        [
                            {
                                "Cell state": new_cell_state,
                                "Can Differentiate Into": new_transition_to,
                                "Publication Found": False,
                                "Selected": False,
                            }
                        ]
                    )
                    st.session_state.df = pd.concat(
                        [st.session_state.df, new_row], ignore_index=True
                    )
                    cell_types = list(
                        set(st.session_state.df["Cell state"]).union(
                            st.session_state.df["Can Differentiate Into"]
                        )
                    )
                    st.success("Row added successfully!")
                    st.rerun()  # Force a rerun to refresh the display in tab1

            # Display and update the table with checkboxes
            st.subheader("Select Paths To Be Found")
            # Update cell_types to include all unique values
            cell_types = list(
                set(st.session_state.df["Cell state"])
                .union(st.session_state.df["Can Differentiate Into"])
                .union(cell_types_to_add)
            )

            for index, row in st.session_state.df.iterrows():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    # Allow users to edit 'Cell state' with a dropdown
                    st.session_state.df.at[index, "Cell state"] = st.selectbox(
                        f"Cell state (row {index})",
                        cell_types,
                        index=cell_types.index(row["Cell state"]),
                    )
                with col2:
                    # Allow users to edit 'Can Differentiate Into' with a dropdown
                    st.session_state.df.at[index, "Can Differentiate Into"] = (
                        st.selectbox(
                            f"Transition To (row {index})",
                            cell_types,
                            index=cell_types.index(row["Can Differentiate Into"]),
                        )
                    )
                with col3:
                    # Checkbox for selection
                    st.session_state.df.at[index, "Selected"] = st.checkbox(
                        "Select", value=row["Selected"], key=f"Selected_{index}"
                    )

            # Select the number of neighbors for the path search
            neighbors_count = st.slider(
                "Select Neighbors Count", min_value=1, max_value=5, value=2
            )

            # Search options
            path_type = st.selectbox(
                "Select Path Type", ["Longest", "Shortest"], key="path_type"
            )

            # Button to trigger path search
            if st.button("Search Existing Paths"):
                # Reset state if path type changes
                st_utils.reset_session_state()

                milestones_top4_metadata = search_path.assign_cells_to_milestones_no_time(
                    st.session_state.cell_data,
                    st.session_state.milestone_percentages,
                    celltype_col=st.session_state.selected_metadata,
                    cell_id_col="cell_id",
                )

                # Filter the DataFrame to get only the rows where 'rank' is 'top1'
                milestones_top1_metadata = milestones_top4_metadata[
                    milestones_top4_metadata["rank"] == "top1"
                ]

                # dimred_milestones and milestones_top1_metadata are your DataFrames
                milestones_top1_metadata = search_path.add_missing_milestones(
                    st.session_state.dimred_milestones,
                    milestones_top1_metadata,
                    st.session_state.selected_metadata,
                )

                # Extract checked rows for the stage order
                checked_rows = st.session_state.df[
                    st.session_state.df["Selected"] == True
                ]
                stage_orders = checked_rows.apply(
                    lambda x: [x["Cell state"], x["Can Differentiate Into"]], axis=1
                ).tolist()

                # Loop through each stage order and perform the search
                for stage_order in stage_orders:
                    # Step 1: Get all paths without filtering
                    path_infos, current_paths = search_path.get_milestones_paths(
                        st.session_state.selected_metadata,
                        st.session_state.milestone_network2,
                        milestones_top1_metadata,
                        stage_order,
                        verbose=False,
                    )

                    # Step 2: Filter paths based on neighbors_count
                    valid_paths, current_valid_paths = search_path.filter_paths_by_neighbors_count(
                        path_infos, current_paths, neighbors_count, verbose=False
                    )

                    # Filter paths based on path_type
                    if path_type == "Shortest":
                        valid_paths, current_valid_paths = search_path.filter_paths_by_length(
                            valid_paths, current_valid_paths, keep_longest=False
                        )
                    else:
                        valid_paths, current_valid_paths = search_path.filter_paths_by_length(
                            valid_paths, current_valid_paths, keep_longest=True
                        )

                    # Generate a string representation of each path
                    for index, (path_info, current_path) in enumerate(
                        zip(valid_paths, current_valid_paths)
                    ):
                        nodes = ", ".join(path_info.get("node", ["N/A"]))
                        stages = ", ".join(path_info.get("stage", ["N/A"]))
                        start_stage = path_info.get("start_stage", "N/A")
                        end_stage = path_info.get("end_stage", "N/A")
                        path_str = (
                            ", ".join(current_path)
                            if isinstance(current_path, list)
                            else current_path
                        )
                        status = "Found" if current_path else "Not Found"

                        # Create path entry with cumulative index
                        path_entry = {
                            "index": st.session_state.cumulative_index,
                            "nodes": nodes,
                            "stages": stages,
                            "start_stage": start_stage,
                            "end_stage": end_stage,
                            "status": status,
                            "path_str": path_str,
                        }

                        # Append to session state
                        st.session_state.all_paths_info.append(path_info)

                        # Append the results to the combined HTML table
                        st.session_state.all_results_table.append(
                            f"<tr><td>{st.session_state.cumulative_index}</td><td>{nodes}</td><td>{stages}</td><td>{start_stage}</td><td>{end_stage}</td><td>{status}</td><td>{path_str}</td></tr>"
                        )

                        # Increment the cumulative index
                        st.session_state.cumulative_index += 1

                # Construct the complete HTML table
                results_html = "<table border='1'><tr><th>Row</th><th>Nodes</th><th>Stages</th><th>Start Stage</th><th>End Stage</th><th>Path Status</th><th>Path</th></tr>"
                results_html += "".join(st.session_state.all_results_table)
                results_html += "</table>"

                # Display the combined HTML table
                st.markdown(results_html, unsafe_allow_html=True)
        else:
            st.warning("Please generate hypothesis.")


    with tab3:
        # Check if all_paths_info has data
        if not st.session_state.get("all_paths_info"):
            st.warning("Please search for paths in Search Path tab.")
        else:
            st.header("Select a path")
            # Display the paths and let the user select one
            options = [
                f"Nodes: {', '.join(entry['node'])}, Stages: {', '.join(entry['stage'])}, Start: {entry['start_stage']}, End: {entry['end_stage']}"
                for entry in st.session_state.all_paths_info
            ]
            selected_index = st.radio(
                "Select a path entry",
                range(len(options)),
                format_func=lambda x: options[x],
            )

            # Show the selected entry details
            selected_entry = st.session_state.all_paths_info[selected_index]
            st.session_state.explain_path_selected_path = selected_entry

            # Button to call the function
            if st.button("Explain Path"):

                fig = create_plot_with_paths(
                    st.session_state.milestone_network2,
                    st.session_state.dimred,
                    st.session_state.dimred_milestones,
                    st.session_state.cell_data,
                    selected_entry,
                    color_by=st.session_state.selected_metadata,
                )

                st.plotly_chart(fig)

                # Call the function with all nodes from path_info as parameters
                explain_path_prompt = prompts.get('Explain path')
                result = explainpath_utils.explain_path(explain_path_prompt, selected_entry)

                # Display the result of the function
                st.success(result)

    with tab4:
    
        # Checks before proceeding
        all_checks_true, stop_string = True, ""

        # List of required variables with their corresponding error messages
        variables_needed = {
            "cell_data": "Cell data is not available.",
            "selected_metadata": "No celltype column selected.",
            "milestone_network2": "Milestone network data is not available.",
            "dimred": "Dimensionality reduction data is not available.",
            "dimred_milestones": "Dimensionality reduction milestones data is not available.",
        }

        # Loop through each required variable and check if it exists in session state and is valid
        for var, error_message in variables_needed.items():
            if var not in st.session_state or st.session_state[var] is None:
                # If any check fails, set all_checks_true to False and set the stop message
                all_checks_true = False
                stop_string = error_message
                break  # Exit loop early on first failure

        # Display warning if any check failed
        if not all_checks_true:
            st.warning(stop_string)
        else:
            st.header("Differential Expression Analysis")
            fig1 = create_plot_nomilestones(
                st.session_state.milestone_network2,
                st.session_state.dimred,
                st.session_state.dimred_milestones,
                st.session_state.cell_data,
                color_by=st.session_state.selected_metadata,
            )

            event = st.plotly_chart(
                fig1, key="342", selection_mode=("box", "lasso"), on_select="rerun"
            )
            button1, button2 = st.columns(2, vertical_alignment="bottom")
            col1, col2 = st.columns([1, 2])
            # Place buttons in the columns
            with col1:
                button1 = st.button("Select Baseline Cells")
            with col2:
                button2 = st.button("Select Comparator Cells")

            # creating a placeholder for the fixed sized textbox
            logtxtbox = st.empty()
            logtxt = "Make 2 selections using the box"
            logtxtbox.text_area("", logtxt, height=10)

            if button1:
                events = event["selection"]["points"]
                if len(events) > 0:
                    cells_1 = st.session_state.cells_1
                    cells_1 = []
                    for i in range(len(events)):
                        cells_1.append(events[i]["text"])
                    st.session_state.cells_1 = cells_1
                    logtxt = "Baseline Cells Selected. Now select comparator cells"
                    logtxtbox.text_area("", logtxt, height=1)
                else:
                    logtxt = "Selection was Empty. Make another selection."
                    logtxtbox.text_area("", logtxt, height=1)

            if button2:
                events = event["selection"]["points"]
                if len(events) > 0:
                    cells_2 = st.session_state.cells_2
                    cells_2 = []
                    for i in range(len(events)):
                        if (
                            events[i]["text"] not in st.session_state.cells_1
                        ):  # avoid overlapping of cells
                            cells_2.append(events[i]["text"])
                    st.session_state.cells_2 = cells_2
                    logtxt = "Comparator Cells Selected"
                    logtxtbox.text_area("", logtxt, height=1)

                else:
                    logtxt = "Selection was Empty. Make another selection."
                    logtxtbox.text_area("", logtxt, height=1)

            if st.session_state.cells_2 is not None:
                if (
                    len(st.session_state.cells_1) > 5
                    and len(st.session_state.cells_2) > 5
                ):
                    logtxt = "Both Selections have been made"
                    logtxtbox.text_area("", logtxt, height=1)
                    button_dge = st.button("Conduct DGE")

                    if button_dge:
                        if st.session_state.cells_1 is not None:
                            cells_1 = st.session_state.cells_1

                        if st.session_state.cells_2 is not None:
                            cells_2 = st.session_state.cells_2

                        if st.session_state.cells_1 is not None:
                            counts_matrix = st.session_state.counts_matrix

                        if len(cells_1) > 10 and len(cells_2) > 10:
                            metad = st.session_state.cell_data
                            dge_df, modified_metadata = dge_utils.conduct_dge(
                                metad, cells_1, cells_2, counts_matrix
                            )
                            st.session_state.dge_df = dge_df
                            st.session_state.modified_metadata = modified_metadata

            if st.session_state.dge_df is not None:
                dge_df = st.session_state.dge_df
                dge_df_display = dge_df[["log2FoldChange", "pvalue", "padj"]]
                st.header("List of Differentially Expressed Genes")
                st.write(dge_df_display)

                button_dge_llm = st.button("Obtain LLM Explanation")
                if button_dge_llm:
                    exp_context = st.session_state.context_input
                    modified_metadata = st.session_state.modified_metadata
                    selected_metadata = st.session_state.selected_metadata
                    time_column = st.session_state.time_column

                    dge_llm_prompt = dge_utils.create_dge_prompt(
                        prompts.get("DGE"),
                        dge_df,
                        modified_metadata,
                        selected_metadata,
                        exp_context,
                        time_column,
                    )
                    dge_llm_res = get_llm_output(
                        st.session_state.selected_llm_models, dge_llm_prompt
                    )
                    st.header("Explanation from LLM")
                    st.success(dge_llm_res)

    with tab5:
        warnings.simplefilter("ignore", category=UserWarning)
        
        cr.settings.verbosity = 2
        sc.settings.set_figure_params(frameon=False, dpi=100)

        # Checks before proceeding
        all_checks_true, stop_string = True, ""

        if "explain_path_selected_path" not in st.session_state:
            st.warning("Please generate hypothesis and select a path.")
        else:
            st.header("Driver Genes")
            # Assuming st.session_state.explain_path_selected_path is already a dictionary
            data = st.session_state.explain_path_selected_path

            # Get start_stage and end_stage
            starting_celltype = data["start_stage"]
            terminal_celltype = data["end_stage"]

            # Get first and last elements of the node list
            first_node = data["node"][0]
            last_node = data["node"][-1]
            starting_cellid = drivergenes_utils.get_highest_percentage_cell(first_node, starting_celltype)
            cellids_in_path = drivergenes_utils.get_cells_for_milestones(data["node"])

            st.session_state.cellids_in_path = cellids_in_path
            st.session_state.starting_cellid = starting_cellid
            st.session_state.start_celltype = starting_celltype
            st.session_state.end_celltype = terminal_celltype

            # List of required variables with their corresponding error messages
            variables_needed = {
                "cell_data": "Cell data is not available.",
                "selected_metadata": "No celltype column selected.",
                "milestone_network2": "Milestone network data is not available.",
                "dimred": "Dimensionality reduction data is not available.",
                "dimred_milestones": "Dimensionality reduction milestones data is not available.",
                "counts_matrix": "Counts matrix is not available",
                "cellids_in_path": "cellids_in_path is not available",
                "starting_cellid": "starting_cellid is not available",
                "end_celltype": "end_celltype is not available",
                "start_celltype": "start_celltype is not available",
            }

            # Loop through each required variable and check if it exists in session state and is valid
            for var, error_message in variables_needed.items():
                if var not in st.session_state or st.session_state[var] is None:
                    # If any check fails, set all_checks_true to False and set the stop message
                    all_checks_true = False
                    stop_string = error_message
                    break  # Exit loop early on first failure

            # Display warning if any check failed
            if not all_checks_true:
                st.warning(stop_string)
            else:
                terminal_celltype = st.session_state.end_celltype
                starting_celltype = st.session_state.start_celltype
                txt1 = (
                    "The starting celltype in the path is "
                    + starting_celltype
                    + " and terminal celltype is "
                    + terminal_celltype
                )
                txt2 = (
                    "Click on the Button below to obtain driver genes to celltype: "
                    + terminal_celltype
                    + ". (Takes ~3-4 minutes for 10k cells)"
                )
                st.write(txt1)
                st.write(txt2)
                button_cellrank = st.button("Obtain Driver Genes", key="24312")

                # if st.session_state.cellrank_drivers_df is None and all_checks_true == True:
                if button_cellrank:
                    # Collect data
                    cell_metadata = st.session_state.cell_data
                    counts_matrix = st.session_state.counts_matrix
                    celltype_column = st.session_state.selected_metadata
                    cellids_in_path = st.session_state.cellids_in_path
                    starting_cellid = st.session_state.starting_cellid
                    terminal_celltype = st.session_state.end_celltype
                    starting_celltype = st.session_state.start_celltype

                    # Normalize Data
                    adata = drivergenes_utils.normalize_data(
                        cell_metadata, counts_matrix, celltype_column, cellids_in_path
                    )

                    # Compute Pseudotime
                    adata_sub, pseudotime_err_msg = drivergenes_utils.compute_pseudotime(
                        adata, starting_cellid
                    )
                    if pseudotime_err_msg != "":
                        stop_string = """
                            CellRank - Pseudotime could not be computed.
                            Here is the Error Message from CellRank:

                            """ + str(
                            pseudotime_err_msg
                        )
                        st.warning(stop_string)
                        st.write("Try selecting a different path")

                    else:
                        print("Finished Computing pseudotime")

                    # Cellrank computation
                    g, err_msg = drivergenes_utils.cellrank_computation(adata_sub, celltype_column)

                    if err_msg != "":
                        stop_string = """
                            CellRank  Macrostates could not be computed.
                            Here is the Error Message from CellRank:

                            """ + str(
                            err_msg
                        )
                        st.warning(stop_string)
                        st.write("Try selecting a different path")

                    else:

                        # Plots for Visualization - cell fates, circular projection
                        folder = os.path.join(script_dir, "../www/images")
                        plots = drivergenes_utils.cellrank_plots(adata_sub, g, celltype_column, folder)
                        
                        # Table for driver genes for a given cell type
                        lineages = list(adata_sub.obsm["lineages_fwd"].names)
                        state = [
                            string for string in lineages if terminal_celltype in string
                        ][0]

                        drivers = g.compute_lineage_drivers(lineages=state)
                        # print(lineages)
                        st.session_state.cellrank_drivers_df = drivers

            if st.session_state.cellrank_drivers_df is not None:

                # Display the table
                terminal_celltype = st.session_state.end_celltype
                table_header = "Driver Genes for Terminal State - " + terminal_celltype
                st.markdown(f"**{table_header}**", unsafe_allow_html=True)
                st.write(st.session_state.cellrank_drivers_df)

                starting_celltype = st.session_state.start_celltype
                col1, col2 = st.columns([2, 1])
                # Display the first image in the first column
                with col1:
                    image_path1 = os.path.join(script_dir, "../www/images/cellrank_pseudotimeplot.png")
                    header1 = "Pseudotime Plot (Reference - " + starting_celltype + ")"
                    st.markdown(f"**{header1}**", unsafe_allow_html=True)
                    st.image(image_path1)

                # Display the second image in the second column
                with col2:
                    image_path2 = os.path.join(script_dir, "../www/images/fateprojections.png")
                    header2 = "Terminal States from CellRank"
                    st.markdown(f"**{header2}**", unsafe_allow_html=True)
                    st.image(image_path2)

                button_cellrank_llm = st.button("Obtain LLM Explanation", key="243")

                if button_cellrank_llm:
                    exp_context = st.session_state.context_input
                    print("exp_context: ", st.session_state.context_input)
                    terminal_celltype = st.session_state.end_celltype
                    cellrank_llm_prompt = drivergenes_utils.create_cellrank_prompt(
                        prompts.get("Cellrank"),
                        st.session_state.cellrank_drivers_df,
                        terminal_celltype,
                        exp_context,
                    )
                    cellrank_llm_res = get_llm_output(
                        st.session_state.selected_llm_models, cellrank_llm_prompt
                    )
                    st.header("Explanation from LLM")
                    st.success(cellrank_llm_res)

    with tab6:
        if st.session_state.dge_df is None:
            st.warning("Please conduct DGE in DGE tab.")
        else:
            st.title("GeneSet Enrichment Analysis")
            col1, col2 = st.columns([1, 2])
            # Place buttons in the columns
            with col1:
                # Define a list of options
                genome_options = ["", "Human", "Mouse", "Rat"]
                # Create a select input box (single selection)
                selected_genome = st.selectbox("Select Genome:", genome_options)
                # button_genome = st.button("Select Genome")
            with col2:
                annot_options = [
                    "",
                    "WikiPathways_2024_Human",
                    "Reactome_2022",
                    "KEGG_2021_Human",
                    "GO_Molecular_Function_2023",
                    "GO_Cellular_Component_2023",
                    "GO_Biological_Process_2023",
                ]
                # Create a select input box (single selection)
                selected_annot = st.selectbox("Select Annotation:", annot_options)

            if (
                selected_genome != ""
                and selected_annot != ""
                and st.session_state.dge_df is not None
            ):
                

                df_sc = st.session_state.dge_df
                df_sc["Rank"] = df_sc["stat"]
                df_sc[["Rank"]] = df_sc[["Rank"]].apply(pd.to_numeric)
                df_sc["Gene"] = df_sc.index
                df_sc["Gene"] = df_sc["Gene"].str.upper()
                df_sc = df_sc[["Gene", "Rank"]]
                ranking = df_sc.sort_values("Rank", ascending=False).reset_index(
                    drop=True
                )
                ranking = ranking.dropna(how="any")
                pre_res = gp.prerank(
                    rnk=ranking, gene_sets=selected_annot, seed=6, permutation_num=100
                )

                gsea_df = gsea_utils.extractProcesses(pre_res)
                gsea_df.sort_values(by=["fdr"], inplace=True)
                gsea_df_display = gsea_df[["Term", "Norm.EnrichmentScore", "fdr"]]
                gsea_df_display["Norm.EnrichmentScore"] = gsea_df_display[
                    "Norm.EnrichmentScore"
                ].round(2)
                gsea_df_display["fdr"] = gsea_df_display["fdr"].round(2)

                # Initialize session state for DataFrame to avoid reinitializing it everytime
                if "gsea_df_display" not in st.session_state:
                    st.session_state.gsea_df_display = pd.DataFrame(gsea_df_display)

                # Set up grid options with row selection enabled
                gb = GridOptionsBuilder.from_dataframe(st.session_state.gsea_df_display)
                gb.configure_selection("multiple", use_checkbox=True)
                grid_options = gb.build()

                # Display the selectable DataFrame
                gsea_aggrid_table = AgGrid(
                    st.session_state.gsea_df_display, gridOptions=grid_options
                )

                # Get the pandas dataframe with only selected rows: will be None when no selected rows
                gsea_aggrid_selected_rows = gsea_aggrid_table["selected_rows"]

                # Get the selected terms
                if gsea_aggrid_selected_rows is not None:
                    selected_terms = gsea_aggrid_selected_rows["Term"].tolist()
                    regulation = gsea_aggrid_selected_rows[
                        "Norm.EnrichmentScore"
                    ].tolist()
                else:
                    selected_terms = []

                # Check there are selected terms
                if len(selected_terms) == 0:
                    st.warning("Select some Terms from the table")
                else:
                    exp_context = st.session_state.context_input
                    metad = st.session_state.modified_metadata
                    selected_metadata = st.session_state.selected_metadata
                    selected_terms_filt = [x.replace(",", " ") for x in selected_terms]

                    llm_fa_prompt = gsea_utils.create_fa_prompt(
                        prompts.get("GSEA"),
                        selected_terms_filt,
                        regulation,
                        metad,
                        exp_context,
                        selected_metadata,
                    )

                    button_fa_llm = st.button("Show LLM Explanation", key="2431244")
                    if button_fa_llm:
                        fa_llm_res = get_llm_output(
                            st.session_state.selected_llm_models, llm_fa_prompt
                        )
                        st.header("Explanation from LLM")
                        st.success(fa_llm_res)

    with tab7:
        # Check there is an LLM model selected
        if st.session_state.selected_llm_models is None:
            st.warning("Please select an LLM model in the Load Dataset tab.")
        else:
            st.header("Enter a claim")

            # Text area for entering a few words on two lines
            claim = st.text_area("Claim:", "")

            # Button to call the function
            if st.button("Verify claim"):
                # Call the function with all nodes from path_info as parameters
                veracity_prompt = prompts.get("Veracity filter")
                result = veracity_utils.veracity_filter(veracity_prompt, claim)
                print(repr(result))
                # Display the result of the function
                st.markdown(result.replace("\n", "  \n"))


if __name__ == "__main__":
    main()
