import pandas as pd
import pydeseq2
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from pydeseq2.utils import load_example_data

def conduct_dge(metad, cells_1, cells_2, counts_matrix):
    
    group_labels = ["group1"] * len(cells_1) + ["group2"] * len(cells_2)
    counts_matrix = counts_matrix.T
    m1 = metad[metad["cell_id"].isin(cells_1)]
    m1["group_label"] = "group_1"
    m2 = metad[metad["cell_id"].isin(cells_2)]
    m2["group_label"] = "group_2"
    metad1 = pd.concat([m1, m2])
    metad1 = metad1.set_index("cell_id")
    cells = cells_1 + cells_2
    counts_matrix = counts_matrix.filter(items=cells, axis=0)
    metad2 = metad1.reindex(counts_matrix.index.to_list())

    # Create DESeqDataSet object
    inference = DefaultInference(n_cpus=1)
    dds = DeseqDataSet(
        counts=counts_matrix,
        metadata=metad2,
        design_factors="group_label",
        # refit_cooks=True,
        inference=inference,
        refit_cooks=True,
        # n_cpus=8, # n_cpus can be specified here or in the inference object
    )

    dds.deseq2()
    stat_res = DeseqStats(dds, inference=inference)
    stat_res.summary()
    stat_res_df = stat_res.results_df
    stat_res_df = stat_res_df.dropna(how="any")
    stat_res_df = stat_res_df.sort_values(by=["padj"])
    return stat_res_df, metad2

def create_dge_prompt(
    dge_prompt, dge_df, metadata, selected_metadata, exp_context, time_column_name
):

    # Genes
    genes = dge_df.index.to_list()
    genes = genes[0:25]
    genes = ", ".join(genes)

    # Log fold changes
    logfcs = list(dge_df["log2FoldChange"])
    logfcs = logfcs[:25]
    logfcs = [str(x) for x in logfcs]
    logfcs = ", ".join(logfcs)

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

    # Time Component
    if time_column_name != "":
        group_1_freq = metadata[metadata["group_label"] == "group_1"][
            time_column_name
        ].value_counts() / len(metadata)
        group_1_time = group_1_freq.index.to_list()
        group_1_pct = group_1_freq * 100
        print(group_1_time)
        group_1_pct = group_1_pct.to_list()
        group_1_pct = ["%.2f" % elem for elem in group_1_pct]
        group_1_time_txt = ""
        for i in range(0, len(group_1_time)):
            if i == len(group_1_time) - 1:
                group_1_time_txt = (
                    group_1_time_txt
                    + str(group_1_pct[i])
                    + "% of cells are at time "
                    + group_1_time[i]
                    + ". "
                )
            else:
                group_1_time_txt = (
                    group_1_time_txt
                    + str(group_1_pct[i])
                    + "% of cell are at time "
                    + group_1_time[i]
                    + ", "
                )

        group_2_freq = metadata[metadata["group_label"] == "group_2"][
            time_column_name
        ].value_counts() / len(metadata)
        group_2_time = group_2_freq.index.to_list()
        group_2_pct = group_2_freq * 100
        print(group_2_time)
        group_2_pct = group_2_pct.to_list()
        group_2_pct = ["%.2f" % elem for elem in group_2_pct]
        group_2_time_txt = ""
        for i in range(0, len(group_2_time)):
            if i == len(group_2_time) - 1:
                group_2_time_txt = (
                    group_2_time_txt
                    + str(group_2_pct[i])
                    + "% of cells are at time "
                    + group_2_time[i]
                    + ". "
                )
            else:
                group_2_time_txt = (
                    group_2_time_txt
                    + str(group_2_pct[i])
                    + "% of cell are at time "
                    + group_2_time[i]
                    + ", "
                )

        print(group_1_time, group_2_time)

    prompt = (
        f"We are comparing 2 regions.  Region 1 has celltypes with their proportions listed within parenthesis ("
        + group_1_celltype_txt
        + "). Region 2 has celltypes with their proportions listed within parenthesis ("
        + group_2_celltype_txt
        + ")."
        + "The top 25 differentially expressed genes between region 1 and region 2 are listed within parenthesis ("
        + genes
        + "). The log2 fold change of the genes are listed within parenthesis ("
        + logfcs
        + ")."
    )

    if time_column_name != "":
        time_prompt = (
            "This is a time series experiment and cells in the 2 regions have different distributions. "
            + "In Region_1, "
            + group_1_time_txt
            + "In Region_2, "
            + group_2_time_txt
            + "Please comment if there was any difference in the regions with respect to time. "
        )
        prompt = prompt + time_prompt

    prompt = prompt + dge_prompt
    if exp_context is not None and len(exp_context) > 0:
        prompt = f"The context of the dataset is within parenthesis ({exp_context}). {prompt}"

    print(prompt)
    return prompt