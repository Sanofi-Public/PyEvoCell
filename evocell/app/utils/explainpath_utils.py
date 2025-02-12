import streamlit as st
from llm.provider import get_llm_output

# Define the function to be called, which processes all the nodes in path_info
def explain_path(explain_path_prompt, path_info):
    # Extract nodes, stages, start_stage, end_stage, and path_str from path_info
    nodes = ", ".join(path_info.get("node", ["N/A"]))
    stages = ", ".join(path_info.get("stage", ["N/A"]))
    start_stage = path_info.get("start_stage", "N/A")
    end_stage = path_info.get("end_stage", "N/A")

    print(
        f"Nodes: {nodes}, Stages: {stages}, Start Stage: {start_stage}, End Stage: {end_stage}"
    )

    prompt = explain_path_prompt + stages
    if (
        st.session_state.context_input is not None
        and len(st.session_state.context_input) > 0
    ):
        prompt = f"The context of the dataset is within parenthesis ({st.session_state.context_input}). {prompt}"

    print(prompt)

    verification = get_llm_output(
        st.session_state.selected_llm_models, prompt
    )  # already prints the output
    return verification