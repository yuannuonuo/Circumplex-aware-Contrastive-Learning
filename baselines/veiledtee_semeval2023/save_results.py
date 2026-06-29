import pandas as pd


def export_results(col_names: list, arguments: list, results: list, filename: str):
    df = pd.DataFrame(columns=col_names)
    df[df.columns[0]] = arguments
    print("\tExporting in Progress")
    for col, result in zip(col_names[1:], results):
        df[col] = result
    df.to_csv(filename, sep='\t', index=False)
    print(f"Results Exported to {filename}!")
