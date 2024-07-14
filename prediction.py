import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the dataset
file_path = '/content/cutoff7.csv'
data = pd.read_csv(file_path)

# Replace '--' with NaN and convert columns to numeric where applicable
data.replace('--', pd.NA, inplace=True)
# All columns from '1G' onwards are rank columns
rank_columns = data.columns[3:]
data[rank_columns] = data[rank_columns].apply(pd.to_numeric, errors='coerce')

# Function to predict next year ranks for a given college and department


def predict_next_year_ranks(data, college_code, dept):
    # Filter data for the given college and department
    filtered_data = data[(data['College_code'] ==
                          college_code) & (data['Dept'] == dept)]

    if filtered_data.empty:
        return pd.DataFrame()

    # Prepare the data for modeling
    X = filtered_data['Year'].values.reshape(-1, 1)
    predictions = {}

    for col in rank_columns:
        y = filtered_data[col].values
        valid_idx = ~np.isnan(y)
        if valid_idx.sum() < 2:  # Need at least two data points to make a prediction
            continue

        X_train, y_train = X[valid_idx], y[valid_idx]

        # Train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict for the next year
        next_year = np.array([[X.max() + 1]])
        prediction = model.predict(next_year)
        predictions[col] = prediction[0]

    if not predictions:
        return pd.DataFrame()

    predictions_df = pd.DataFrame(predictions, index=[0])
    predictions_df.insert(0, 'Dept', dept)
    predictions_df.insert(0, 'College_code', college_code)
    predictions_df.insert(0, 'Year', X.max() + 1)
    return predictions_df


# Predict ranks for all colleges and departments
all_colleges = data['College_code'].unique()
all_predictions = pd.DataFrame()

for college in all_colleges:
    unique_departments = data[data['College_code'] == college]['Dept'].unique()
    for dept in unique_departments:
        dept_predictions = predict_next_year_ranks(data, college, dept)
        all_predictions = pd.concat(
            [all_predictions, dept_predictions], ignore_index=True)

# Save the predictions to a new CSV file
output_file_path = '/content/predicted_ranks.csv'
all_predictions.to_csv(output_file_path, index=False)

# print(f"Predictions saved to {output_file_path}")
