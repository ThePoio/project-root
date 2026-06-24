import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(config: dict):
    """
    Carga el dataset, lo procesa y lo divide en train/test.
    """
    df = pd.read_csv(config['data_loader']['raw_data_path'])

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df = df.drop(columns=['customerID'])

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data_loader']['test_size'], 
        random_state=config['data_loader']['random_state'], 
        stratify=y
    )

    median_train = X_train['TotalCharges'].median()
    X_train['TotalCharges'] = X_train['TotalCharges'].fillna(median_train)
    X_test['TotalCharges'] = X_test['TotalCharges'].fillna(median_train)

    mapping_gender = {'Female': 1, 'Male': 0}
    mapping_yes_no = {'Yes': 1, 'No': 0}
    X_train['gender'] = X_train['gender'].map(mapping_gender)
    X_test['gender'] = X_test['gender'].map(mapping_gender)
    X_train['Partner'] = X_train['Partner'].map(mapping_yes_no)
    X_test['Partner'] = X_test['Partner'].map(mapping_yes_no)
    y_train = y_train.map(mapping_yes_no)
    y_test = y_test.map(mapping_yes_no)

    return X_train, X_test, y_train, y_test