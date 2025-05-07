import pandas as pd
import os


def export_analyze_3_to_latex(df, method="tabpfn", output_path="C:/Users/Minu/OneDrive/Arbeit/HTWG/Master/Masterarbeit/thesis_teamprojekt_templates-master/chapters/baseline_mean_crps_nll_diff_feature_models.tex"):
    def escape_underscores(text):
        return text.replace("_", r"\_") if isinstance(text, str) else text

    def df_to_latex(df, table_label, caption):
        df.columns = [r"\mbox{" + escape_underscores(col) + "}" for col in df.columns]
        df.index = [escape_underscores(str(idx)) for idx in df.index]

        col_format = "l" + "X" * df.shape[1]
        header = "Feature & " + " & ".join(df.columns) + r" \\"

        latex = "\\begin{table}[htbp!]\n\\centering\n\\small\n"
        latex += "\\begin{tabularx}{\\textwidth}{" + col_format + "}\n"
        latex += "\\toprule\n" + header + "\n\\midrule\n"

        for idx, row in df.iterrows():
            latex += idx + " & " + " & ".join(map(str, row)) + r" \\" + "\n"

        latex += "\\bottomrule\n\\end{tabularx}\n"
        latex += f"\\caption{{{caption}}}\n\\label{{table:{table_label}}}\n\\end{{table}}\n"
        return latex

    # Choose caption based on method
    if method == "tabpfn":
        caption = "Mean CRPS and NLL of different feature models using TabPFN for Q4 2022 training."
    else:
        caption = "Mean CRPS and NLL of different feature models using the baseline model for Q4 2022 and 2016–2022 training."

    latex_content = df_to_latex(df, f"{method}_mean_scores_diff_feature_models", caption)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"LaTeX table snippet saved to: {output_path}")



def export_analyze_baseline_for_different_feature_models(result_df, result_df_ft, score="nll"):

    if score == "nll":
        output_path="C:/Users/Minu/OneDrive/Arbeit/HTWG/Master/Masterarbeit/thesis_teamprojekt_templates-master/chapters/baseline_nll_full_diff_feature_models.tex"
    else:
        output_path="C:/Users/Minu/OneDrive/Arbeit/HTWG/Master/Masterarbeit/thesis_teamprojekt_templates-master/chapters/baseline_crps_full_diff_feature_models.tex"
    
    def escape_underscores(text):
        return text.replace("_", r"\_") if isinstance(text, str) else text

    def df_to_latex(df, table_label, caption):
        df.columns = [escape_underscores(col) for col in df.columns]
        df.index = [escape_underscores(str(idx)) for idx in df.index]

        col_format = "l" + "X" * df.shape[1]
        header = "Feature & " + " & ".join(df.columns) + " \\\\"

        latex = f"\\begin{{table}}[htbp!]\n\\centering\n\\scriptsize\n"
        latex += "\\begin{adjustbox}{max width=\\textwidth}\n"
        latex += "\\begin{tabularx}{\\textwidth}{" + col_format + "}\n"
        latex += "\\toprule\n" + header + "\n\\midrule\n"

        for idx, row in df.iterrows():
            latex += escape_underscores(idx) + " & " + " & ".join(map(str, row)) + " \\\\\n"

        latex += "\\bottomrule\n\\end{tabularx}\n"
        latex += "\\end{adjustbox}\n"
        latex += f"\\caption{{{caption}}}\n\\label{{table:{table_label}}}\n\\end{{table}}\n\n"
        return latex


    crps_caption = f"{score} scores for the baseline model for different feature models (Q4 training)."
    nll_caption = f"{score} scores for the baseline model for different feature models (2016-2022 training)."

    latex_content = df_to_latex(result_df, f"baseline_full_{score}_q4", crps_caption)
    latex_content += df_to_latex(result_df_ft, f"baseline_full_{score}_ft", nll_caption)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"LaTeX document has been saved to: {output_path}")


def export_baseline_parameters_for_different_feature_models(
    baseline_parameters_q4,
    baseline_parameters_ft,
    output_path="C:/Users/Minu/OneDrive/Arbeit/HTWG/Master/Masterarbeit/thesis_teamprojekt_templates-master/chapters/baseline_parameters_diff_feature_models.tex"
):
    def escape_underscores(text):
        if not isinstance(text, str):
            return text
        return text.replace("\\_", "TEMPESC").replace("_", r"\_").replace("TEMPESC", r"\_")

    def format_column_name(col):
        if isinstance(col, str) and col.startswith("beta_"):
            # Extract the suffix after 'beta_' and split by underscores
            suffix = col.split("_")[1:]
            # Join parts by commas instead of underscores
            formatted_suffix = ",".join(suffix)
            # Format as LaTeX math mode
            return f"$\\beta_{{{formatted_suffix}}}$"
            # Handle sigma_sq case (e.g., sigma_sq)
        elif col == "sigma_sq":
                return "$\\sigma^2$"
        return escape_underscores(col)

    def df_to_latex(df, table_label, caption):
        formatted_columns = [format_column_name(col) for col in df.columns]
        df.index = [escape_underscores(str(idx)) for idx in df.index]

        # Use "p" column for feature name, and centered X for each beta
        col_format = "p{3.2cm}" + ">{\\centering\\arraybackslash}X" * df.shape[1]
        header = "Feature & " + " & ".join(formatted_columns) + " \\\\"  # Ensure proper header formatting

        latex = "\\begin{table}[htbp!]\n\\centering\n\\small\n"
        latex += "\\resizebox{\\textwidth}{!}{%\n"
        latex += "\\begin{tabularx}{\\textwidth}{" + col_format + "}\n"
        latex += "\\toprule\n" + header + "\n\\midrule\n"

        for idx, row in df.iterrows():
            latex += escape_underscores(idx) + " & " + " & ".join(map(str, row)) + " \\\\\n"

        latex += "\\bottomrule\n\\end{tabularx}\n"
        latex += "}%\n"
        latex += f"\\caption{{{caption}}}\n\\label{{table:{table_label}}}\n\\end{{table}}\n\n"
        return latex

    baseline_parameters_q4_caption = (
        "Estimated model parameters for baseline regression models using different feature "
        "configurations for Q4 training. The $\\beta_1$ coefficients are reordered to place "
        "'power\\_t-96' first."
    )

    baseline_parameters_ft_caption = (
        "Estimated model parameters for baseline regression models using different feature "
        "configurations for 2016--2022 training. The $\\beta_1$ coefficients are reordered to place "
        "'power\\_t-96' first."
    )

    latex_content = df_to_latex(
        baseline_parameters_q4, "baseline_parameters_q4", baseline_parameters_q4_caption
    )
    latex_content += df_to_latex(
        baseline_parameters_ft, "baseline_parameters_ft", baseline_parameters_ft_caption
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)

    print(f"LaTeX document has been saved to: {output_path}")





def export_final_comparison_to_latex(crps_matrix, nll_matrix, output_path="C:/Users/Minu/OneDrive/Arbeit/HTWG/Master/Masterarbeit/thesis_teamprojekt_templates-master/chapters/overall_comparison_of_crps_nll.tex"):
    def escape_underscores(text):
        return text.replace("_", r"\_") if isinstance(text, str) else text

    def df_to_latex(df, table_label, caption):
        df.columns = [escape_underscores(col) for col in df.columns]
        df.index = [escape_underscores(str(idx)) for idx in df.index]

        col_format = "l" + "X" * df.shape[1]
        header = "Feature & " + " & ".join(df.columns) + " \\\\"

        latex = f"\\begin{{table}}[htbp!]\n\\centering\n\\small\n"
        latex += "\\begin{tabularx}{\\textwidth}{" + col_format + "}\n"
        latex += "\\toprule\n" + header + "\n\\midrule\n"

        for idx, row in df.iterrows():
            latex += escape_underscores(idx) + " & " + " & ".join(map(str, row)) + " \\\\\n"

        latex += "\\bottomrule\n\\end{tabularx}\n"
        latex += f"\\caption{{{caption}}}\n\\label{{table:{table_label}}}\n\\end{{table}}\n\n"
        return latex

    latex_content = r"""
\documentclass{article}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{caption}
\usepackage{float}
\usepackage[margin=1in]{geometry}
\begin{document}
"""

    crps_caption = "Comparative Continuous Ranked Probability Score (CRPS) across different forecasting models and feature representations. Results are shown for models trained on Q4 data and full-year (2016) data."
    nll_caption = "Comparative Negative Log-Likelihood (NLL) performance across different forecasting models and feature representations. Columns reflect models trained on Q4 data versus full-year (2016) training data."

    latex_content += df_to_latex(crps_matrix, "crps_comparison", crps_caption)
    latex_content += df_to_latex(nll_matrix, "nll_comparison", nll_caption)
    latex_content += r"\end{document}"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"LaTeX document has been saved to: {output_path}")



def export_scores_id_1_to_latex(nll_df, crps_df, output_path="C:/Users/Minu/OneDrive/Arbeit/HTWG/Master/Masterarbeit/thesis_teamprojekt_templates-master/chapters/analysis_1_scores_id_1.tex"):
    # Helper function to escape underscores
    def escape_underscores(text):
        if isinstance(text, str):
            return text.replace("_", r"\_")
        return text
    # Define a helper function to convert dataframe to LaTeX table format
    def df_to_latex(df, table_label, caption):
        # Escape underscores in column names
        df.columns = [escape_underscores(col) for col in df.columns]

        latex = f"\\begin{{table}}[htbp!]\n"
        latex += f"\\centering\n"
        latex += f"\\begin{{tabular}}{{" + "l" + "c" * (df.shape[1] - 1) + "}\n"
        latex += f"\\hline\n"
        latex += " & ".join(df.columns) + " \\\\ \n"
        latex += f"\\hline\n"

        # Escape underscores in DataFrame values
        for i, row in df.iterrows():
            latex += " & ".join([escape_underscores(str(value)) for value in row]) + " \\\\ \n"
        latex += f"\\hline\n"
        latex += f"\\end{{tabular}}\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{table:{table_label}}}\n"
        latex += f"\\end{{table}}\n\n"
        return latex

    # Create the LaTeX document content
    latex_content = r"\documentclass{article}" + "\n"
    latex_content += r"\usepackage{graphicx}" + "\n"
    latex_content += r"\usepackage{longtable}" + "\n"
    latex_content += r"\usepackage{amsmath}" + "\n"
    latex_content += r"\begin{document}" + "\n"

    # Add tables for nll_df and crps_df
    latex_content += df_to_latex(nll_df, "table_tabpfn_analysis_1_nll", "NLL Scores for Experiment ID 1 (22Q1 Training 23Q1 Validation). " \
    "Shown are the mean, median, minimum and max values obtained by different calculation methods.")  # For nll_df table
    latex_content += df_to_latex(crps_df, "table_tabpfn_analysis_1_crps", "CRPS Scores for Experiment ID 1 (22Q1 Training 23Q1 Validation). " \
    "Shown are the mean, median, minimum and max values obtained by different calculation methods.")  # For crps_df table

    # End the document
    latex_content += r"\end{document}"

    # Save the LaTeX content to a file
    with open(output_path, 'w') as f:
        f.write(latex_content)

    print(f"Latex document has been saved to: {output_path}")

def export_analysis_2_result_tables_to_latex(crps_matrix, nll_matrix, chapter_tex_path):
    """
    Exports CRPS and NLL result matrices from analysis_2 into a LaTeX document with formatted tables.
    
    The LaTeX document will contain two tables: one for CRPS results and one for NLL results, 
    with appropriate captions and labels for referencing in a larger LaTeX document.
    """
    # Generate LaTeX table strings
    crps_latex = crps_matrix.to_latex(float_format="%.3f", na_rep="-", index=True)
    nll_latex = nll_matrix.to_latex(float_format="%.3f", na_rep="-", index=True)
    # Write full standalone LaTeX document
    with open(chapter_tex_path, "w", encoding="utf-8") as f:
        f.write(r"""\documentclass[a4paper,12pt]{article}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1in}
\usepackage{caption}
\usepackage{amsmath}

\title{TabPFN Result Tables}
\author{}
\date{}

\begin{document}
\maketitle
""")
        f.write(r"\section*{CRPS Results}" + "\n")
        f.write(r"\begin{table}[ht]\centering" + "\n")
        f.write(crps_latex + "\n")
        f.write(r"\caption{CRPS values obtained by TabPFN for different combinations of training and validation date ranges. The feature model is always $P_{t-96}$, and the two mean wind speeds.}" + "\n")
        f.write(r"\label{tab:crps}" + "\n")
        f.write(r"\end{table}" + "\n\n")

        f.write(r"\section*{NLL Results}" + "\n")
        f.write(r"\begin{table}[ht]\centering" + "\n")
        f.write(nll_latex + "\n")
        f.write(r"\caption{NLL values obtained by TabPFN for different combinations of training and validation date ranges. The feature model is always $P_{t-96}$, and the two mean wind speeds.}" + "\n")
        f.write(r"\label{tab:nll}" + "\n")
        f.write(r"\end{table}" + "\n")

        f.write(r"\end{document}")
