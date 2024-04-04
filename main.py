import pandas as pd
import numpy as np
import read_data
import data_analysis
import data_visualization
import os
import shutil

if __name__ == '__main__':
    # set pandas options
    read_data.set_pandas_options()

    # create results folder
    # Check if the folder exists
    if os.path.exists('results'):
        # If the folder exists, delete it
        shutil.rmtree('results')
        print(f"'{'results'}' folder exists. Deleting and recreating it.")
    else:
        print(f"'{'results'}' folder does not exist. Creating it.")
    # Create the folder
    os.makedirs('results')
    print(f"'{'results'}' folder has been created.")

    # read and inspect dataset
    dataset_file_path = '/home/weifengzhou/SKEMA_Business_School/Big Data and applied data analytics in supply chain ' \
                        'management - Common Resources/Big Data Team Project/archive/ONLINE EDUCATION SYSTEM REVIEW.csv'
    # encoding: replace the string values in the dataset by numerical values.
    df_data = read_data.read_and_inspect_data(path=dataset_file_path)
    # store the data description after encoding
    df_numeric = data_analysis.replace_strings_in_dataset_with_integer(df_data)
    with pd.ExcelWriter('results/data_info.xlsx', mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
        df_numeric.describe().to_excel(writer, sheet_name='DescribeAfterEncoding')

    # analyze data
    # 1. analyze the importance of different factors to "Performance in online"
    data_analysis.analyze_important_factors_to_performance_in_online(df_numeric)
    # 2. analyze all correlations
    data_analysis.analyze_correlation_matrix(df_numeric)

    # data visualization
    # 1. plot "Average marks scored before pandemic in traditional classroom"
    data_visualization.plot_marks_scored_before_pandemic_in_traditional_classroom(df_numeric)
    # 2. plot "Performance in online"
    data_visualization.plot_performance_in_online(df_data)
    # 3. plot "Level of Education"
    data_visualization.plot_level_of_education(df_data)
    # 4. plot "Device type used to attend classes"
    data_visualization.plot_device_types(df_data)
    # 5. plot the correlation matrix
    data_visualization.plot_correlation_matrix()

    print("Results!!!: The data analysis results have been written into the folder \"results\"")
