import os
import pandas as pd

if __name__ == '__main__':
    dp_df = pd.read_csv("../../raw/06.utf8.csv", delimiter=';', index_col=False)

    diag_columns = [col for col in dp_df.columns if 'DIA_' in col]
    proc_columns = [col for col in dp_df.columns if 'PROC_' in col]

    diag_rows = []
    proc_rows = []
    for _, row in dp_df.iterrows():
        for diag_column in diag_columns:
            if not pd.isna(row[diag_column]):
                diag_rows.append({'patientid': row['PATIENT ID'], 'diag': row[diag_column]})
        for proc_column in proc_columns:
            if not pd.isna(row[proc_column]):
                proc_rows.append({'patientid': row['PATIENT ID'], 'proc': row[proc_column]})

    diag_df = pd.DataFrame(diag_rows)
    diag_df_counts = diag_df.groupby(by='patientid').size().reset_index(name='counts')

    proc_df = pd.DataFrame(proc_rows)
    proc_df_counts = proc_df.groupby(by='patientid').size().reset_index(name='counts')

    print(f"{diag_df_counts['counts'].mean()} & {proc_df_counts['counts'].mean()}")
    print(f"{diag_df_counts['counts'].std()} & {proc_df_counts['counts'].std()}")
    print(f"{diag_df_counts['counts'].median()} & {proc_df_counts['counts'].median()}")
    print(f"{diag_df_counts['counts'].max() - diag_df_counts['counts'].min()} & {proc_df_counts['counts'].max() - proc_df_counts['counts'].min()}")
    print(f"{diag_df_counts['counts'].min()} & {proc_df_counts['counts'].min()}")
    print(f"{diag_df_counts['counts'].max()} & {proc_df_counts['counts'].max()}")
    print(f"{diag_df_counts['counts'].sum()} & {proc_df_counts['counts'].sum()}")
    print(f"{diag_df_counts['counts'].shape[0]} & {proc_df_counts['counts'].shape[0]}")
    print(f"{len(set(diag_df['diag']))} & {len(set(proc_df['proc']))}")
