from llm.provider import get_llm_output
from llm.process_hypothesis import process_paragraph_for_veracity
import streamlit as st

def veracity_filter(veracity_prompt, claim):
    prompt = veracity_prompt + claim
    if (
        st.session_state.context_input is not None
        and len(st.session_state.context_input) > 0
    ):
        prompt = f"The context of the dataset is within parenthesis ({st.session_state.context_input}). {prompt}"

    print(prompt)

    verification = get_llm_output(st.session_state.selected_llm_models, prompt)

    # process answer from LLM
    is_true, is_valid, citations = process_paragraph_for_veracity(
        verification, progress_callback=None
    )

    if is_true is True:
        if is_valid:
            result = "Claim is valid.\n\n" + "\n\n".join(citations)
        else:
            result = "Claim is Plausible, but cannot find citations in Pubmed."
    else:
        result = "Claim cannot be verified."

    print(result)
    return result