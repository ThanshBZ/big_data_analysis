import pandas as pd
import numpy as np
import read_data
from scipy.stats import pointbiserialr
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def replace_strings_in_dataset_with_integer(df):
    df_tmp = pd.DataFrame()
    for column in df.select_dtypes(exclude=['object']).columns:
        df_tmp[column] = df[column]

    df_tmp['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df_tmp['Home Location'] = df['Home Location'].map({'Rural': 0, 'Urban': 1})
    df_tmp['Level of Education'] = df['Level of Education'].map({'School': 0, 'Under Graduate': 1, 'Post Graduate': 2})
    df_tmp['Device type used to attend classes'] = \
        df['Device type used to attend classes'].map({'Mobile': 0, 'Laptop': 1, 'Desktop': 2})
    df_tmp['Economic status'] = df['Economic status'].map({'Poor': 0, 'Middle Class': 1, 'Rich': 2})
    df_tmp['Are you involved in any sports?'] = df['Are you involved in any sports?'].map({'No': 0, 'Yes': 1})
    df_tmp['Do elderly people monitor you?'] = df['Do elderly people monitor you?'].map({'No': 0, 'Yes': 1})
    df_tmp['Interested in Gaming?'] = df['Interested in Gaming?'].map({'No': 0, 'Yes': 1})
    df_tmp['Have separate room for studying?'] = df['Have separate room for studying?'].map({'No': 0, 'Yes': 1})
    df_tmp['Engaged in group studies?'] = df['Engaged in group studies?'].map({'No': 0, 'yes': 1})
    df_tmp['Average marks scored before pandemic in traditional classroom'] = \
        df['Average marks scored before pandemic in traditional classroom'].map({
            '0-10': 1,
            '11-20': 2,
            '21-30': 3,
            '31-40': 4,
            '41-50': 5,
            '51-60': 6,
            '61-70': 7,
            '71-80': 8,
            '81-90': 9,
            '91-100': 10
        })
    df_tmp['Interested in?'] = df['Interested in?'].map({'Theory': 0, 'Practical': 1, 'Both': 2})
    df_tmp['Your level of satisfaction in Online Education'] = \
        df['Your level of satisfaction in Online Education'].map({'Bad': 0, 'Average': 1, 'Good': 2})

    return df_tmp


def analyze_coorelation_between_home_location_and_internet_facility(df):
    # Assuming 'Internet facility in your locality' is numerical
    correlation, p_value = pointbiserialr(df['Home Location'], df['Internet facility in your locality'])
    # print(f"\nPoint Biserial Correlation: {correlation}, P-value: {p_value}")

    correlation_df = pd.DataFrame({
        "Correlation": [correlation],
        "P-value": [p_value]
    }).reset_index(drop=True)
    with pd.ExcelWriter('results/coorelation_between_home_location_and_internet_facility.xlsx', engine='openpyxl') \
            as writer:
        correlation_df.to_excel(writer, sheet_name='correlation')


def analyze_important_factors_to_performance_in_online(df):
    # Splitting the dataset into features and target variable
    X = df.drop('Performance in online', axis=1)
    y = df['Performance in online']

    # Split the data into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=10, min_samples_leaf=10)

    # Perform cross-validation
    cv_scores = cross_val_score(rf_regressor, X, y, cv=5, scoring='r2')  # 5-fold cross-validation

    # Print the cross-validation scores
    print("Cross-validation R^2 scores:", cv_scores)
    print("Mean cross-validation R^2 score:", cv_scores.mean())

    # Fit the model on the training data
    rf_regressor.fit(X, y)

    # Predict on the test data
    # y_pred = rf_regressor.predict(X_test)
    #
    # # Evaluate the model
    # mse = mean_squared_error(y_test, y_pred)
    # print(f"Mean Squared Error: {mse}")

    # Get feature importances
    feature_importances = rf_regressor.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(
        by='Importance', ascending=False)

    # Save the feature importances to an Excel file
    with pd.ExcelWriter('results/feature_importance_to_performance_in_online.xlsx', engine='openpyxl') as writer:
        feature_importance_df.to_excel(writer, sheet_name='feature_importance', index=False)

    # # Initialize the XGBoost Regressor
    # xgb_regressor = XGBRegressor(n_estimators=100, random_state=42, learning_rate=0.1,
    #                              objective='reg:squarederror', max_depth=3, min_child_weight=10)
    #
    # # Perform cross-validation
    # cv_scores = cross_val_score(xgb_regressor, X, y, cv=5, scoring='r2')
    #
    # # Convert scores to positive values and print
    # print("Cross-validation R^2 scores:", cv_scores)
    # print("Mean cross-validation R^2 score:", cv_scores.mean())
    #
    # # Fit the model on the entire dataset
    # xgb_regressor.fit(X, y)
    #
    # # Get feature importances
    # feature_importances = xgb_regressor.feature_importances_
    # feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(
    #     by='Importance', ascending=False)
    #
    # # Save the feature importances to an Excel file
    # with pd.ExcelWriter('results/feature_importance_to_performance_in_online.xlsx', engine='openpyxl') as writer:
    #     feature_importance_df.to_excel(writer, sheet_name='feature_importance', index=False)


def analyze_correlation_matrix(df):
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
    sorted_coor_results.to_excel("results/correlation_matrix.xlsx", index=False, engine='openpyxl')
