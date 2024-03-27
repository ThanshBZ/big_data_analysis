import matplotlib

matplotlib.use('TkAgg')  # Use TkAgg for PyCharm compatibility. Adjust if necessary.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import read_data
from scipy.stats import pointbiserialr
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder


def replace_values_or_strings(data_array, list_original_values, list_final_values):
    if len(list_original_values) != len(list_final_values):
        print(
            "Error!!!: The lengths of \"list_original_values\" and \"list_final_values\" are not equal. Please check!")
        read_data.get_error_info()
        exit(-3)

    # Using a dictionary for mapping original values to replacement values for efficiency
    replacement_dict = dict(zip(list_original_values, list_final_values))

    # Replace values for numpy arrays
    if isinstance(data_array, np.ndarray):
        # Vectorized replacement using numpy
        return np.vectorize(replacement_dict.get)(data_array, data_array)
    # Replace values for lists or other iterable types
    else:
        # Using list comprehension for replacements
        return [replacement_dict.get(item, item) for item in data_array]


def plot_marks_scored_before_pandemic_in_traditional_classroom(df):
    # replace values
    x = np.array(df['Average marks scored before pandemic in traditional classroom'])
    list_original_values = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80',
                            '81-90', '91-100']
    list_final_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    xx = replace_values_or_strings(x, list_original_values, list_final_values)
    # create a temporary dataframe
    df_copy = pd.DataFrame({"Average marks scored before pandemic in traditional classroom": xx}).reset_index(drop=True)

    plt.figure(figsize=(18, 12))
    sns.set(style='darkgrid')
    sns.countplot(x='Average marks scored before pandemic in traditional classroom', data=df_copy, palette='Dark2',
                  hue='Average marks scored before pandemic in traditional classroom', legend=False)
    plt.ylim(0, 350)
    plt.title('Average marks scored before pandemic in traditional classroom', size=20)
    plt.xlabel('Average marks scored before pandemic in traditional classroom', weight='bold', fontsize=18)
    plt.ylabel('Number of students', weight='bold', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig("results/Average marks scored before pandemic in traditional classroom.png")
    # plt.show()


def plot_performance_in_online(df):
    plt.figure(figsize=(18, 12))
    sns.set(style='darkgrid')
    sns.countplot(x='Performance in online', data=df, palette='Dark2',
                  hue='Performance in online', legend=False)
    plt.ylim(0, 350)
    plt.title('Performance in online', size=20)
    plt.xlabel('Performance in online', weight='bold', fontsize=18)
    plt.ylabel('Number of students', weight='bold', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
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
    plt.savefig("results/Level of Education.png")


def analyze_coorelation_between_home_location_and_internet_facility(df):
    df_tmp = pd.DataFrame()
    df_tmp['Home Location'] = df['Home Location'].map({'Urban': 1, 'Rural': 0})
    df_tmp['Internet facility in your locality'] = df['Internet facility in your locality']

    # Assuming 'Internet facility in your locality' is numerical
    correlation, p_value = pointbiserialr(df_tmp['Home Location'], df_tmp['Internet facility in your locality'])
    # print(f"\nPoint Biserial Correlation: {correlation}, P-value: {p_value}")

    correlation_df = pd.DataFrame({
        "Correlation": [correlation],
        "P-value": [p_value]
    }).reset_index(drop=True)
    with pd.ExcelWriter('results/coorelation_between_home_location_and_internet_facility.xlsx', engine='openpyxl') \
            as writer:
        correlation_df.to_excel(writer, sheet_name='correlation')


def plot_device_types(df):
    plt.figure(figsize=(18, 12))
    sns.countplot(x='Device type used to attend classes', data=df, palette='Dark2',
                  hue='Device type used to attend classes', legend=False)
    plt.title('Device type used to attend classes', weight='bold', size=20)
    plt.xlabel('Device type used to attend classes', weight='bold', fontsize=18)
    plt.ylabel('Number of students', weight='bold', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig("results/Device type used to attend classes.png")


def analyze_important_factors_to_performance_in_online(df):
    label_encoder = LabelEncoder()
    label_mappings = {}
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
        label_mappings[column] = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(label_mappings)
    # print(df.head())

    # Splitting the dataset into training and testing sets
    X = df.drop('Performance in online', axis=1)
    y = df['Performance in online']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing and training the Decision Tree Regressor
    dt_regressor = DecisionTreeRegressor(random_state=42)
    dt_regressor.fit(X_train, y_train)

    # Feature importance
    feature_importance = pd.Series(dt_regressor.feature_importances_, index=X.columns).sort_values(ascending=False)
    feature_importance_df = pd.DataFrame(feature_importance).reset_index()
    feature_importance_df.columns = ['Feature', 'Importance']

    with pd.ExcelWriter('results/feature_importance_to_performance_in_online.xlsx', engine='openpyxl') \
            as writer:
        feature_importance_df.to_excel(writer, sheet_name='feature_importance')


def analyze_important_factors_to_internet_facility(df):
    label_encoder = LabelEncoder()
    label_mappings = {}
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = label_encoder.fit_transform(df[column])
        label_mappings[column] = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(label_mappings)


    correlation_data = []

    columns = df.columns
    for i, var1 in enumerate(columns):
        for var2 in columns[i + 1:]:
            if var1 == var2:
                continue  # Skip comparing the variable to itself
            # Compute Pearson correlation and p-value, ensuring no NaN values
            corr, p_value = pearsonr(df[var1].dropna(), df[var2].dropna())

            correlation_data.append({
                'Variable 1': var1,
                'Variable 2': var2,
                'Correlation Coefficient': corr,
                'P-value': p_value
            })

    # Convert the list of dictionaries to a DataFrame
    correlation_results = pd.DataFrame(correlation_data)
    sorted_coor_results = \
        correlation_results.sort_values(by='Correlation Coefficient', ascending=False).reset_index(drop=True)
    # print(sorted_coor_results)
    sorted_coor_results.to_excel("results/correlation_matrix.xlsx", index=True, engine='openpyxl')
