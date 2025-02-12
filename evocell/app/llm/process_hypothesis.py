from .provider import get_llm_output
from .pubmed_api import paper_exists
import re
import html
import sys
import requests
from xml.etree import ElementTree
import csv
import queue

def update_progress(message, progress_queue):
    print("update_progress called")
    print(message)
    progress_queue.put(message)  # Put the message into the queue

# Function to process hypothesis data in a separate thread
def process_hypothesis_data_in_thread(
    llm_model, answer_differentiation_table: str, results, progress_queue
):
    try:
        # Process the hypothesis data
        headers, processed_rows = process_hypothesis_data(
            llm_model,
            answer_differentiation_table,
            progress_callback=lambda msg: update_progress(msg, progress_queue)
        )

        # Store results in the local 'results' dictionary
        results["headers"] = headers
        results["processed_rows"] = processed_rows

        # Debug to confirm local variables are updated
        print("Local variables updated with headers and processed_rows.")

    except Exception as e:
        # Print any errors that occur within the thread
        print(f"An error occurred in thread processing: {e}")

def process_hypothesis_data(
    llm_model, answer_differentiation_table: str, progress_callback=None
):
    """
    Parses and processes the input CSV to extract transitions and verify their validity.

    Returns a tuple containing:
    - headers: The headers of the table.
    - processed_rows: A list of tuples (cell_state, target_state, is_true, is_valid, citation).
    """
    print("start process_hypothesis_data()")
    answer_differentiation_table = format_hypothesis_str(answer_differentiation_table)
    print(answer_differentiation_table)

    # Get headers and rows
    elements = answer_differentiation_table.split("\n")
    headers = elements[0].split(",")
    rows = [element.split(",") for element in elements[1:]]
    rows = [row for row in rows if len(row) == 2 and all(row)]

    processed_rows = []
    seen_rows = set()

    # Calculate the total number of transitions to process
    total_transitions = sum(
        len(cells[1].strip('"').replace("\\", "").split(","))
        for cells in rows
        if cells[0] and cells[1] and cells[0] != "NA" and cells[1] != "NA"
    )
    transitions_checked = 0

    # First pass: process the rows to expand and clean data
    for cells in rows:
        cell_state = cells[0].strip('"').replace("\\", "")
        transition_to = cells[1].strip('"').replace("\\", "")

        # Skip rows with empty cell state, transition_to, containing "NA", "None", or None
        if (
            not cell_state
            or cell_state in {"NA", "None", "No", None}
            or not transition_to
            or transition_to in {"NA", "None", "No", None}
        ):
            continue

        # Split the transition_to field if it contains multiple states
        target_states = [state.strip() for state in transition_to.split(",")]

        # Create a row entry for each individual target state
        for target_state in target_states:
            print(cell_state)
            print(target_state)

            transitions_checked += 1  # Increment count of checked transitions

            # Notify progress
            if progress_callback:
                progress_callback(
                    f"Checking transition {transitions_checked}/{total_transitions} from '{cell_state}' to '{target_state}' with LLM and pubmed"
                )

            # Verify the validity of the transition
            is_true, is_valid, citation = verify(
                llm_model, cell_state, target_state, progress_callback
            )
            row_tuple = (cell_state, target_state, is_true, is_valid, citation)

            # Add the row only if it has not been seen before
            if row_tuple not in seen_rows:
                processed_rows.append(row_tuple)
                seen_rows.add(row_tuple)
    
    # Filter out the transitions that are not valid and remove column
    processed_rows = [
        (cell_state, target_state, is_true, citation) 
        for cell_state, target_state, is_true, is_valid, citation in processed_rows
        if is_valid  # Filter based on 'is_valid' value
    ]

    print("headers:")
    print(headers)
    print("processed_rows:")
    print(processed_rows)
    # Add the new column "Publication Found"
    headers.append("Publication Found")
    return headers, processed_rows

def verify(llm_model, first_cell, second_cell, progress_callback=None):
    # hypo_gen_prompt_prefix = "Is the following claim valid? Answer Yes or no and give three paper titles to support your answer. Don't include the author or any other information, just the titles and please make sure the paper titles exist. Please answer in the exact format here:\nYes/No\nTitle1:\nTitle2:\nTitle3:\n\nHere is the claim:"
    # prompt = hypo_gen_prompt_prefix + first_cell + " differentiates into " + second_cell

    prompt = f"Do {first_cell} cells transition to {second_cell} cells? Do {first_cell} cells differentiate to {second_cell} cells? Please give response as yes or no only. If yes, then retrieve 3 complete titles of articles from pubmed and make sure the paper titles exist. Output the results in the exact format here:\n Yes/No\nTitle1:\nTitle2:\nTitle3:\n\n"
    # Optionally update progress if callback is provided
    # if progress_callback:
    #    progress_callback(f"Generating paper titles with LLM")
    print(prompt)
    verification = get_llm_output(llm_model, prompt)
    print(verification)
    # Call process_paragraph_for_hypogen with progress callback
    return process_paragraph_for_hypogen(verification, progress_callback)


def process_paragraph_for_hypogen(paragraph, progress_callback=None):
    paragraph = paragraph.replace("<br>", " ")

    # Regex to match a title followed by a digit
    pattern = r"(Title \d+:|Title\d+:)"

    # Split the paragraph using the regex pattern, keeping the initial part
    parts = re.split(pattern, paragraph)
    first_part = parts[0].strip()

    yes = "yes"
    if yes not in first_part.lower():
        return False, False, None   # transition is not possible

    for i in range(1, len(parts), 2):
        this_title = parts[i + 1].strip()
        # Optionally update progress for each paper checked
        # if progress_callback:
        #    progress_callback(f"Checking existence of paper '{this_title}' on PubMed")

        citation = paper_exists(this_title)
        if citation:
            return True, True, citation # transition possible and backed by pubmed paper

    return True, False, False # transition possible but not backed by paper

def process_paragraph_for_veracity(paragraph, progress_callback=None):
    paragraph = paragraph.replace("<br>", " ")

    # Regex to match a title followed by a digit
    pattern = r"(Title \d+:|Title\d+:)"

    # Split the paragraph using the regex pattern, keeping the initial part
    parts = re.split(pattern, paragraph)
    first_part = parts[0].strip()

    yes = "yes"
    if yes not in first_part.lower():
        return False, False, None   # transition is not possible

    valid_citations = False
    real_citations = []
    for i in range(1, len(parts), 2):
        this_title = parts[i + 1].strip()
        # Optionally update progress for each paper checked
        # if progress_callback:
        #    progress_callback(f"Checking existence of paper '{this_title}' on PubMed")

        citation = paper_exists(this_title)
        if citation:
            valid_citations = True
            real_citations.append(citation)

    if len(real_citations)>0:
        return True, valid_citations, real_citations

    return True, False, None # transition possible but not backed by paper

def process_paragraph_for_hypogen_old(paragraph):
    paragraph = paragraph.replace("<br>", " ")
    # print("paragraph after processed",paragraph)
    # Regex to match a title followed by a digit
    pattern = r"(Title \d+:|Title\d+:)"

    # Split the paragraph using the regex pattern, keeping the initial part
    parts = re.split(pattern, paragraph)
    # print("parts",parts)
    # Combine initial part and title-content pairs
    combined_parts = [parts[0].strip()]
    first_part = combined_parts[0]
    yes = "yes"
    no = "no"
    if yes not in first_part.lower():
        return False, None

    for i in range(1, len(parts), 2):
        this_title = parts[i + 1].strip()
        # print("this_title",this_title)
        citation = paper_exists(this_title)
        if citation:
            return True, citation
    return False, None


def verify_old(llm_model, first_cell, second_cell):
    # hypo_gen_prompt_prefix = "Is the following claim valid? Answer Yes or no and give three paper titles to support your answer. Don't include the author or any other information, just the titles and please make sure the paper titles exist. Please answer in the exact format here:\nYes/No\nTitle1:\nTitle2:\nTitle3:\n\nHere is the claim:"
    hypo_gen_prompt_prefix = "give me complete title for publications for pubmed that describe transition or differentiation of cells from "
    prompt = (
        hypo_gen_prompt_prefix
        + first_cell
        + " to "
        + second_cell
        +"."
    )
    verification = get_llm_output(llm_model, prompt)
    return process_paragraph_for_hypogen(verification)


def process_string(input_string):
    # Split the input string by <br>
    substrings = input_string.split("<br>")

    # Regex pattern to match |string_A|string_B| format
    pattern = re.compile(r"^\|([^|]+)\|([^|]+)\|$")

    # Lists to store the processed substrings and citations
    processed_substrings = []
    citations = []
    footnote_counter = 1
    processed_table = []
    table_header = "<tr><th>Initial Cell State</th><th>Transition To</th></tr>"
    processed_table.append(table_header)
    for substring in substrings:
        match = pattern.match(substring.strip())
        if match:
            string_A = match.group(1)
            string_B = match.group(2)
            # if "initial" in string_A.lower() or string_A=="-":
            #     processed_substrings.append(substring.strip())
            # else:
            is_true, is_valid, citation = verify(string_A, string_B)
            if is_valid:
                footnote_text = f"<tr><td>{string_A}</td><td>{string_B}[{footnote_counter}]</td></tr>"
                processed_table.append(footnote_text)
                citations.append(f"[{footnote_counter}] {citation}")
                footnote_counter += 1
        else:
            if (
                "initial" not in substring
                and "-" not in substring
                and "table" not in substring
            ):
                processed_substrings.append(substring.strip())
            else:
                processed_substrings.append(substring.strip())

    # Combine processed substrings and add citations at the end
    result_string = "\n".join(processed_substrings)
    table = "".join(processed_table)
    table = "<table border='1'>" + table
    table = table + "</table>"
    result_string = result_string + table
    if citations:
        result_string += "\n\n" + "\n".join(citations)
    html_str = result_string.replace("\n", "<br>")
    return html_str


def pubmed_api(paragraph: str):
    paragraph = paragraph.replace("<br>", " ")
    # print("paragraph after processed",paragraph)
    # Regex to match a title followed by a digit
    pattern = r"(Title \d+:|Title\d+:)"

    # Split the paragraph using the regex pattern, keeping the initial part
    parts = re.split(pattern, paragraph)
    # print("parts",parts)
    # Combine initial part and title-content pairs
    combined_parts = [parts[0].strip()]
    first_part = combined_parts[0]
    yes = "yes"
    no = "no"
    if yes in first_part.lower():
        result = [
            "Yes, your claim is valid. Here are some papers to support my result."
        ]
    elif no in first_part.lower():
        result = [
            "No, your claim is not valid. Here are some papers to support my result."
        ]
    else:
        return paragraph

    with_backup = False
    for i in range(1, len(parts), 2):
        this_title = parts[i + 1].strip()
        # print("this_title",this_title)
        citation = paper_exists(this_title)
        if citation:
            with_backup = True
            result.append(f'\n\nPubMed citation: "{citation}"')
    if not with_backup:
        result = ["Sorry, I do not have enough information to verify your conjecture"]

    processed_paragraph = "".join(result)
    processed_paragraph = html.escape(processed_paragraph)
    processed_paragraph = processed_paragraph.replace("\n", "<br>")
    return processed_paragraph


def format_hypothesis_str(answer_differentiation_table: str) -> str:
    """
    Formats the string for hypothesis generation

    Args:
        answer_differentiation_table (str): The original CSV string.

    Returns:
        str: The formatted CSV string with one target state per row.
    """
    print("start format_hypothesis_str()")
    print("answer_differentiation_table is:")
    print(repr(answer_differentiation_table))
    print("\n")

    formatted_table = (
        "Cell state:Can Differentiate Into;" + answer_differentiation_table
    )
    formatted_table = formatted_table.replace(":", ",").replace(";", "\n")

    print("the formatted_table is:")
    print(repr(formatted_table))
    return formatted_table


def format_csv(answer_differentiation_table: str) -> str:
    """
    Formats the CSV string to ensure that entries in the 'Can Differentiate Into' column
    are split into separate rows for each target state, removes any backslashes, and ensures each target state is a single word.

    Args:
        answer_differentiation_table (str): The original CSV string.

    Returns:
        str: The formatted CSV string with one target state per row.
    """
    print("start format_csv()")
    print("answer_differentiation_table is:")
    print(repr(answer_differentiation_table))
    processed_lines = []

    # Read the CSV input
    csv_reader = csv.reader(answer_differentiation_table.strip().split("\n"))
    headers = next(csv_reader)  # Extract headers
    processed_lines.append(headers)  # Keep headers as-is

    # Process each row to split 'Can Differentiate Into' entries
    for cells in csv_reader:
        # Ensure the row is not empty and has at least 2 elements
        if len(cells) < 2 or not cells[0].strip() or not cells[1].strip():
            continue

        # Remove backslashes from the cell state and transition data
        cell_state = cells[0].strip('"').replace("\\", "")
        transition_data = ",".join(cells[1:]).strip('"').replace("\\", "")
        target_states = [state.strip() for state in transition_data.split(",")]

        # Create a new row for each target state, ensuring no spaces within states
        for target_state in target_states:
            # Skip empty, wildcard target states, or target states containing spaces
            if target_state and target_state != "*" and " " not in target_state:
                processed_lines.append([cell_state, target_state])

    # Convert processed lines back to CSV format
    formatted_table = "\n".join(
        [f'"{line[0]}","{line[1]}"' for line in processed_lines if len(line) == 2]
    )

    print("the formatted_table is:")
    print(repr(formatted_table))
    return formatted_table

def generate_html_table(headers, processed_rows):
    """
    Generates an HTML table from processed rows and appends footnotes for valid transitions.

    Returns the HTML string for the table and citations.
    """
    # Prepare the initial HTML table header
    processed_table = []

    processed_table.append(
        "Here is a table of potential cell state transitions, ordered from most likely to less likely based on the given cell types:<br><br>"
    )

    table_header = f"<tr><th>{headers[0]}</th><th>{headers[1]}</th></tr>"
    processed_table.append(table_header)

    citations = []
    footnote_counter = 1

    # Generate the HTML table rows
    for cell_state, target_state, citation in processed_rows:
        if citation:
            # Include a footnote for valid transitions
            footnote_text = f"<tr><td>{cell_state}</td><td>{target_state} [{footnote_counter}]</td></tr>"
            citations.append(f"[{footnote_counter}] {citation}")
            footnote_counter += 1
        else:
            # No footnote for invalid transitions
            footnote_text = f"<tr><td>{cell_state}</td><td>{target_state} [no publication found]</td></tr>"

        processed_table.append(footnote_text)

    # Combine the processed table into a complete HTML table
    table = "".join(processed_table)
    table_html = f"<table border='1'>{table}</table>"

    # Combine citations if any
    if citations:
        citations_html = "<br>".join(citations)
        result_string = table_html + "<br><br>" + citations_html
    else:
        result_string = table_html

    # Replace newlines with HTML breaks for formatting
    html_str = result_string.replace("\n", "<br>")
    return html_str


def process_hypothesis(llm_model, headers, processed_rows):
    headers, processed_rows = process_hypothesis_data(
        llm_model, answer_differentiation_table
    )
    html_output = generate_html_table(headers, processed_rows)
    return html_output


def process_hypothesis_old(headers, processed_rows):
    """
    Generates an HTML table from processed rows and appends footnotes for valid transitions.

    Returns the HTML string for the table and citations.
    """
    # Prepare the initial HTML table header
    processed_table = []
    table_header = f"<tr><th>{headers[0]}</th><th>{headers[1]}</th></tr>"
    processed_table.append(table_header)

    citations = []
    footnote_counter = 1

    # Generate the HTML table rows
    for cell_state, target_state, citation in processed_rows:
        if citation:
            # Include a footnote for valid transitions
            footnote_text = f"<tr><td>{cell_state}</td><td>{target_state} [{footnote_counter}]</td></tr>"
            citations.append(f"[{footnote_counter}] {citation}")
            footnote_counter += 1
        else:
            # No footnote for invalid transitions
            footnote_text = f"<tr><td>{cell_state}</td><td>{target_state}</td></tr>"

        processed_table.append(footnote_text)

    # Combine the processed table into a complete HTML table
    table = "".join(processed_table)
    table_html = f"<table border='1'>{table}</table>"

    # Combine citations if any
    if citations:
        citations_html = "<br>".join(citations)
        result_string = table_html + "<br><br>" + citations_html
    else:
        result_string = table_html

    # Replace newlines with HTML breaks for formatting
    html_str = result_string.replace("\n", "<br>")
    return html_str


def process_hypothesis_old2(answer_differentiation_table: str):
    # Parse the CSV input properly to handle commas within quotes
    csv_reader = csv.reader(answer_differentiation_table.strip().split("\n"))
    headers = next(csv_reader)  # Extract headers
    rows = list(csv_reader)  # Extract rows

    # List to hold fully processed rows before generating HTML
    processed_rows = []

    # First pass: process the rows to expand and clean data
    for cells in rows:
        cell_state = cells[0].strip('"')
        transition_to = cells[1].strip('"')

        # Skip rows with empty cell state or transition_to
        if not cell_state or not transition_to:
            continue

        # Split the transition_to field if it contains multiple states
        target_states = [state.strip() for state in transition_to.split(",")]

        # Create a row entry for each individual target state
        for target_state in target_states:
            processed_rows.append((cell_state, target_state))

    # Prepare the processed table list with the initial HTML table header
    processed_table = []
    table_header = f"<tr><th>{headers[0]}</th><th>{headers[1]}</th></tr>"
    processed_table.append(table_header)

    citations = []
    footnote_counter = 1
    print(processed_rows)
    # Second pass: generate the HTML table rows and verify transitions
    for cell_state, target_state in processed_rows:
        # Verify the validity of the transition (assuming you have a `verify` function)
        is_valid, citation = verify(cell_state, target_state)
        if is_valid:
            # Include a footnote if there is a valid transition
            footnote_text = f"<tr><td>{cell_state}</td><td>{target_state} [{footnote_counter}]</td></tr>"
            citations.append(f"[{footnote_counter}] {citation}")
            footnote_counter += 1
            processed_table.append(footnote_text)

    # Combine the processed table into a complete HTML table
    table = "".join(processed_table)
    table_html = f"<table border='1'>{table}</table>"

    # Combine citations if any
    if citations:
        citations_html = "<br>".join(citations)
        result_string = table_html + "<br><br>" + citations_html
    else:
        result_string = table_html

    # Replace newlines with HTML breaks for formatting
    html_str = result_string.replace("\n", "<br>")
    return html_str


if __name__ == "__main__":
    inputstr = sys.argv[1]
    output_str = process_string(inputstr)
    print(output_str)
