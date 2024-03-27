import pandas as pd
import inspect


def get_error_info():
    # Get the current frame
    current_frame = inspect.currentframe()
    # Go back to the frame of the caller
    caller_frame = inspect.getouterframes(current_frame, 2)[1]

    # Extracting the function name
    # Fallback for global script execution context
    function_name = caller_frame.function if caller_frame.function != '<module>' else __name__

    print('=========== get_error_info: beginning ===========')
    print(f"File Name: {caller_frame.filename}")  # File name
    print(f"Function Name: {function_name}")  # Function name or module name if directly executed
    print(f"Line No: {caller_frame.lineno}")  # Line number
    print('=========== get_error_info: end ===========\n')


def set_pandas_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)


def read_input_file(path=''):
    if not path:
        print('Error!!!: the input of the following module is incorrect. Please check!')
        get_error_info()
        exit(-1)

    df = pd.read_csv(path)

    return df


def inspect_data(df, path='results/data_info.xlsx'):
    print("\n=========== Data inspection summary: beginning ===========")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": df.notnull().sum(),
        "Null Count": df.isnull().sum(),
        "Nunique Count": df.nunique(),
        "Dtype": df.dtypes
    }).reset_index(drop=True)

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        info_df.to_excel(writer, sheet_name='Info')
        df.describe().to_excel(writer, sheet_name='Describe')

    is_dataset_ok = True
    # Check for missing values in each column
    missing_values_count = df.isnull().sum()
    # Check if there are any missing values in the DataFrame
    if missing_values_count.any():
        # Raise an alarm (print a message for this example)
        print("Alert: Missing values found in the dataset!")
        # Optionally, print the columns with the count of missing values
        print(missing_values_count[missing_values_count > 0])
        is_dataset_ok = False
    else:
        print("No missing values found.")

    # Find duplicated rows, keeping 'False' for the first occurrence and 'True' for subsequent duplicates
    duplicated_rows_mask = df.duplicated(keep='first')
    # Check if there are any duplicated rows
    if duplicated_rows_mask.any():
        print("Alert: Duplicated rows found in the dataset!")
        # Get the index (row numbers) of the duplicated rows
        duplicated_row_indices = df.index[duplicated_rows_mask].tolist()
        print(f"Row numbers of duplicated rows: {duplicated_row_indices}")
        is_dataset_ok = False
    else:
        print("No duplicated rows found.")

    print("=========== Data inspection summary: end ===========\n")

    if not is_dataset_ok:
        get_error_info()
        exit(-2)


def read_and_inspect_data(path=''):
    df = read_input_file(path=path)

    inspect_data(df)

    return df


if __name__ == '__main__':
    set_pandas_options()

    input_file_path = '/home/weifengzhou/SKEMA_Business_School/Big Data and applied data analytics in supply chain ' \
                      'management - Common Resources/Big Data Team Project/archive/ONLINE EDUCATION SYSTEM REVIEW.csv'
    df_data = read_input_file(path=input_file_path)

    inspect_data(df_data)
