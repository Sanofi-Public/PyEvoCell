import streamlit as st
import pandas as pd

# Initialize session state
def initialize_session():
    if "selected_llm_models" not in st.session_state:
        st.session_state.selected_llm_models = None

    if "headers" not in st.session_state:
        st.session_state.headers = None

    if "processed_rows" not in st.session_state:
        st.session_state.processed_rows = None

    if "cell_data" not in st.session_state:
        st.session_state.cell_data = None

    if "selected_metadata" not in st.session_state:
        st.session_state.selected_metadata = None

    if "milestone_network2" not in st.session_state:
        st.session_state.milestone_network2 = None

    if "all_paths" not in st.session_state:
        st.session_state.all_paths = None

    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame(
            columns=[
                "Cell state",
                "Can Differentiate Into",
                "Publication Found",
                "Selected",
            ]
        )

    if "previous_input_dir" not in st.session_state:
        st.session_state.previous_input_dir = None

    if "context_input" not in st.session_state:
        st.session_state.context_input = None

    if "time_column" not in st.session_state:
        st.session_state.time_column = None

    if "cells_1" not in st.session_state:
        st.session_state.cells_1 = None

    if "cells_2" not in st.session_state:
        st.session_state.cells_2 = None

    if "dge_df" not in st.session_state:
        st.session_state.dge_df = None

    if "cellrank_drivers_df" not in st.session_state:
        st.session_state.cellrank_drivers_df = None

    if "modified_metadata" not in st.session_state:
        print("modified_metadata initialized")
        st.session_state.modified_metadata = None

# Function to reset session state when search parameters change
def reset_session_state():
    st.session_state.all_paths_info = []
    st.session_state.all_results_table = []
    st.session_state.cumulative_index = 1

