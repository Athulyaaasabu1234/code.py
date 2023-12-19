import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
import sklearn.cluster as cluster
import sklearn.metrics as skmet

# Scaling the data using MinMaxScaler
def scale_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

# Function to plot inertia vs the number of clusters for k-means
def elbow_plot(data):
    inertias = []
    for x in range(1, 11):
        kmeans = cluster.KMeans(n_clusters=x)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    plt.plot(range(1, 11), inertias, marker='.')
    plt.title('Elbow plot')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

# Function to find the index of the first country in a given cluster
def find_index(clusters, n):
    return next(i for i, cluster in enumerate(clusters) if cluster == n)

# Function to fit a polynomial curve to the data and make predictions
def curve_objective(x, *coefficients):
    return np.polyval(coefficients, x)

def fit_and_predict_co2_curve(X, Y, degree):
    # Fit the polynomial curve
    coefficients = np.polyfit(X, Y, degree)

    # Plotting the original data and the fitted curve
    plt.scatter(X, Y, label='Original Data')
    fitted_curve = np.polyval(coefficients, X)
    plt.plot(X, fitted_curve, 'r--', label=f'Fitted Curve (Degree {degree})')
    plt.xlabel('Year')
    plt.ylabel('CO2 Emissions in kt')
    plt.title('Fitted Curve for CO2 Emissions Over Time')
    plt.legend()
    plt.show()

    # Predicting CO2 emissions for future years
    future_years = [2025, 2030, 2035, 2040, 2045]
    predictions = [np.polyval(coefficients, year) for year in future_years]
    for year, prediction in zip(future_years, predictions):
        print(f'The predicted CO2 emissions in Australia in {year} is {prediction:.2f} kt.')

# Function to plot k number of clusters using k-means clustering algorithm
def plot_kmeans(data, n):
    kmeans = cluster.KMeans(n_clusters=n)
    kmeans.fit(data)
    cen = kmeans.cluster_centers_

    # Printing the centers of the clusters
    for i in range(n):
        print(f'The coordinates of the center of cluster {i+1} are ({cen[i, 0]}, {cen[i, 1]})')

    # Printing the silhouette score
    print('The Silhouette score of the clusters is ', skmet.silhouette_score(data, kmeans.labels_))

    # Plotting the scaled clusters
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
    plt.xlabel('Scaled values of the amount of CO2 emitted')
    plt.ylabel('Scaled values of the forest area')
    plt.title('K-means clustering')
    for i in range(n):
        plt.plot(cen[i, 0], cen[i, 1], "*", markersize=10, c='g')
    plt.show()

    return kmeans

# Function to compare samples of different clusters
def compare_the_CO2_data_among_countries_in_various_clusters(data, first_country, second_country):
    print(data.iloc[[first_country, second_country], :])

# Load data from CSV files
co2_data = pd.read_csv('co2_emission_data.csv')
forest_data = pd.read_csv('forest_area.csv')

# Combine data into a single dataframe
data = pd.DataFrame({
    'Country': co2_data.iloc[:, 0].values,
    'Amount of CO2 emitted': co2_data.iloc[:, 63],
    'Forest Area': forest_data.iloc[:, 63]
})

# Drop rows with missing values
data = data.dropna()
data_before_scaling = data

# Scale the data
data = scale_data(data.iloc[:, 1:])

# Determine the value of k using the elbow method
elbow_plot(data)

# Use k-means clustering algorithm to determine and plot the clusters
kmeans = plot_kmeans(data, 2)

# Compare two countries' data belonging to two different clusters
country_from_first_cluster = find_index(kmeans.labels_, 0)
country_from_second_cluster = find_index(kmeans.labels_, 1)
compare_the_CO2_data_among_countries_in_various_clusters(data_before_scaling, country_from_first_cluster, country_from_second_cluster)

# Selecting only the data from row 8 and columns 34 to 54 for CO2 emissions
co2_data_row = co2_data.iloc[13, 34:64].values.astype(float)

# Using the range of years as the independent variable
years = np.array(range(1990, 2020))

# Call the function with corrected parameters
fit_and_predict_co2_curve(years, co2_data_row, degree=4)  
