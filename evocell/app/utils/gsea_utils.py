import pandas as pd
import gseapy as gp

def create_fa_prompt(gsea_prompt, selected_terms, reg, metadata, exp_context, selected_metadata):
    # Extracting the top 5 celltypes and their percentages
    group_1_freq = (
        metadata[metadata["group_label"] == "group_1"][selected_metadata].value_counts()
        / len(metadata)
    )[0:5]
    group_1_celltypes = group_1_freq.index.to_list()
    group_1_pct = group_1_freq * 100
    print(group_1_celltypes)
    group_1_pct = group_1_pct.to_list()
    group_1_pct = ["%.2f" % elem for elem in group_1_pct]
    group_1_celltype_txt = ""
    for i in range(0, len(group_1_celltypes)):
        if i == len(group_1_celltypes) - 1:
            group_1_celltype_txt = (
                group_1_celltype_txt
                + "and "
                + group_1_celltypes[i]
                + " is "
                + str(group_1_pct[i])
                + "%. "
            )
        else:
            group_1_celltype_txt = (
                group_1_celltype_txt
                + group_1_celltypes[i]
                + " is "
                + str(group_1_pct[i])
                + "%, "
            )

    group_2_freq = (
        metadata[metadata["group_label"] == "group_2"][selected_metadata].value_counts()
        / len(metadata)
    )[0:5]
    group_2_celltypes = group_2_freq.index.to_list()
    group_2_pct = group_2_freq * 100
    print(group_2_celltypes)
    group_2_pct = group_2_pct.to_list()
    group_2_pct = ["%.2f" % elem for elem in group_2_pct]
    group_2_celltype_txt = ""
    for i in range(0, len(group_2_celltypes)):
        if i == len(group_2_celltypes) - 1:
            group_2_celltype_txt = (
                group_2_celltype_txt
                + "and "
                + group_2_celltypes[i]
                + " is "
                + str(group_2_pct[i])
                + "%. "
            )
        else:
            group_2_celltype_txt = (
                group_2_celltype_txt
                + group_2_celltypes[i]
                + " is "
                + str(group_2_pct[i])
                + "%, "
            )

    print(group_1_celltypes, group_2_celltypes)

    terms = ", ".join(selected_terms)

    regulation_txt = ""
    for i in range(0, len(selected_terms)):
        if reg[i] > 0:
            regulation_txt = (
                regulation_txt
                + " "
                + selected_terms[i]
                + " is up regulated in Region_2, "
            )
        else:
            regulation_txt = (
                regulation_txt
                + " "
                + selected_terms[i]
                + " is down regulated in Region_2, "
            )

    print(regulation_txt)

    group_1_freq = metadata[metadata["group_label"] == "group_1"][
        selected_metadata
    ].value_counts()[:5]
    group_1_celltypes = group_1_freq.index.to_list()
    group_1_celltypes = ", ".join(group_1_celltypes)
    group_2_freq = metadata[metadata["group_label"] == "group_2"][
        selected_metadata
    ].value_counts()[:5]
    group_2_celltypes = group_2_freq.index.to_list()
    group_2_celltypes = ", ".join(group_2_celltypes)
    print(group_1_celltypes, group_2_celltypes)

    pre_prompt = (
        "We are comparing 2 regions. Region 1 consists of the celltypes listed within parenthesis ("
        + group_1_celltypes
        + "). Region 2 consists of celltypes listed within parenthesis ("
        + group_2_celltypes
        + ")."
        + "Geneset enrichment analysis between Region_1 and Region_2 show that, "
        + regulation_txt
        + ". "
    )
    prompt = pre_prompt + gsea_prompt
    if exp_context is not None and len(exp_context) > 0:
        prompt = f"The context of the dataset is within parenthesis ({exp_context}). {prompt}"

    print(prompt)
    return prompt

def extractProcesses(pre_res, fdr_cutoff=1):
    """
    Extracts Terms from Pre-res object
    """
    out = []
    for term in list(pre_res.results):
        out.append(
            [
                term,
                pre_res.results[term]["fdr"],
                pre_res.results[term]["es"],
                pre_res.results[term]["nes"],
            ]
        )
        out_df = pd.DataFrame(out, columns=["Term", "fdr", "es", "nes"])

    out_df = out_df.sort_values(by=["fdr"], ascending=True)
    out_df.rename(
        columns={
            "nes": "Norm.EnrichmentScore",
        },
        inplace=True,
    )
    out_df.reset_index(drop=True, inplace=True)
    return out_df