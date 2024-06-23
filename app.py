from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# app = Flask(__name__, static_url='/static')

app = Flask(__name__)
app.config['STATIC_URL'] = '/static'

# Load data
data_path = './cutoff-2021.csv'
college_names_path = './Collegewithcode.csv'

data = pd.read_csv(data_path)
college_names = pd.read_csv(college_names_path)

# Replace "--" with NaN and convert to numeric
data.replace("--", np.nan, inplace=True)
for col in data.columns[2:]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Function to get college recommendations


def get_college_recommendations(rank, category, department, data, college_names, n_neighbors=10):
    # Filter data for the given department
    dept_data = data[data['Dept'] == department]

    # Drop rows with NaN values in the given category column
    dept_data = dept_data.dropna(subset=[category])

    # Prepare the data for k-NN
    X = dept_data[[category]].values

    # Fit the k-NN model
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X)

    # Find the nearest neighbors
    distances, indices = knn.kneighbors([[rank]])

    # Get the closest colleges
    closest_colleges = dept_data.iloc[indices[0]]

    # Sort the colleges by their cutoff ranks
    sorted_colleges = closest_colleges.sort_values(by=category)

    # Split into two groups: less likely and more likely
    less_likely = sorted_colleges[sorted_colleges[category] < rank].tail(15)
    more_likely = sorted_colleges[sorted_colleges[category] >= rank].head(15)

    # Merge with college names
    less_likely = less_likely.merge(
        college_names, on='College_code', how='left')
    more_likely = more_likely.merge(
        college_names, on='College_code', how='left')

    return less_likely[['College_code', 'College_name']].values, more_likely[['College_code', 'College_name']].values


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    rank = int(request.form['rank'])
    category = request.form['category']
    department = request.form['department']

    less_likely_colleges, more_likely_colleges = get_college_recommendations(
        rank, category, department, data, college_names)

    return render_template('recommend.html', less_likely_colleges=less_likely_colleges, more_likely_colleges=more_likely_colleges)


if __name__ == '__main__':
    app.run(debug=True)
