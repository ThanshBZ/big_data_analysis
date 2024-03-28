import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg for PyCharm compatibility. Adjust if necessary.

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_marks_scored_before_pandemic_in_traditional_classroom(df):
    plt.figure(figsize=(18, 12))
    sns.set(style='darkgrid')
    sns.countplot(x='Average marks scored before pandemic in traditional classroom', data=df, palette='Dark2',
                  hue='Average marks scored before pandemic in traditional classroom', legend=False)
    plt.ylim(0, 350)
    plt.title('Average marks scored before pandemic in traditional classroom', weight='bold', size=20)
    plt.xlabel('Average marks scored before pandemic in traditional classroom', weight='bold', fontsize=18)
    plt.ylabel('Number of students', weight='bold', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("results/Average marks scored before pandemic in traditional classroom.png")
    # plt.show()


def plot_performance_in_online(df):
    plt.figure(figsize=(18, 12))
    sns.set(style='darkgrid')
    sns.countplot(x='Performance in online', data=df, palette='Dark2',
                  hue='Performance in online', legend=False)
    plt.ylim(0, 350)
    plt.title('Performance in online', weight='bold', size=20)
    plt.xlabel('Performance in online', weight='bold', fontsize=18)
    plt.ylabel('Number of students', weight='bold', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("results/Performance in online.png")


def plot_level_of_education(df):
    plt.figure(figsize=(18, 12))
    sns.countplot(x='Level of Education', data=df, palette='Dark2',
                  hue='Level of Education', legend=False)
    plt.title('Level of Education', weight='bold', size=20)
    plt.xlabel('Level of Education', weight='bold', fontsize=18)
    plt.ylabel('Number of students', weight='bold', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("results/Level of Education.png")


def plot_device_types(df):
    plt.figure(figsize=(18, 12))
    sns.countplot(x='Device type used to attend classes', data=df, palette='Dark2',
                  hue='Device type used to attend classes', legend=False)
    plt.title('Device type used to attend classes', weight='bold', size=20)
    plt.xlabel('Device type used to attend classes', weight='bold', fontsize=18)
    plt.ylabel('Number of students', weight='bold', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("results/Device type used to attend classes.png")


def plot_correlation_matrix():
    # # Read the Excel file
    file_path = 'results/correlation_matrix.xlsx'
    df = pd.read_excel(file_path)

    # Define a mapping from variable names to numeric indexes
    df_var = pd.read_excel('results/data_info.xlsx', sheet_name='Info')
    all_vars = np.array(df_var['Column'])
    # all_vars = np.union1d(df['Variable 1'], df['Variable 2'])
    var_to_idx = {var: idx for idx, var in enumerate(all_vars)}
    idx_to_var = {idx: var for var, idx in var_to_idx.items()}

    # Initialize an empty correlation matrix and a mask using numeric indexes
    corr_matrix = pd.DataFrame(index=range(len(all_vars)), columns=range(len(all_vars)), data=np.nan)
    mask_matrix = pd.DataFrame(index=range(len(all_vars)), columns=range(len(all_vars)), data=True)

    # Fill in the correlation matrix and update the mask using numeric indexes
    p_value_threshold = 0.05
    for _, row in df.iterrows():
        idx1, idx2 = var_to_idx[row['Variable 1']], var_to_idx[row['Variable 2']]
        corr, p_value = row['Correlation Coefficient'], row['P-value']
        if p_value < p_value_threshold:  # If the correlation is significant
            corr_matrix.loc[idx1, idx2] = corr
            corr_matrix.loc[idx2, idx1] = corr  # Make the matrix symmetric
            mask_matrix.loc[idx1, idx2] = False
            mask_matrix.loc[idx2, idx1] = False

    # Plot the correlation matrix
    plt.figure(figsize=(18, 12))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", mask=mask_matrix,
                linewidths=.5, cbar=True, square=True, vmin=-1, vmax=1)
    plt.title('Correlation Matrix with P-value < ' + str(p_value_threshold), weight='bold', size=20)

    # Adjust the color bar tick label font size
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)  # Set font size of the scale numbers

    # Adjust the ticks to show numeric indexes
    plt.xticks(ticks=np.arange(len(all_vars)) + 0.5, labels=range(len(all_vars)), rotation=90, size=16)
    plt.yticks(ticks=np.arange(len(all_vars)) + 0.5, labels=range(len(all_vars)), rotation=0, size=16)

    plt.xlabel('Attribute indexes', weight='bold', fontsize=18)
    plt.ylabel('Attribute indexes', weight='bold', fontsize=18)

    plt.tight_layout()
    plt.savefig("results/Indexed Correlation Matrix with P-value Threshold.png")
    # plt.show()

    # # Optionally, print or save the index-variable mapping for reference
    # print("Variable Index Mapping:")
    # print(idx_to_var)


