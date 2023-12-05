import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    # Read CSV file into a python dataframe, skipping the first 4 rows
    df = pd.read_csv(filename, skiprows=4)

    # Select relevant columns (Country Name and columns P to BM)
    df_selected = df.iloc[[4, 7, 9, 33, 83], [0] + list(range(35, 65))]

    # Transpose the dataframe and set 'Country Name' as the index
    df_transposed = df_selected.set_index('Country Name').transpose()

    # Clean the transposed dataframe
    df_transposed.columns.name = 'Year'
    df_transposed.index = pd.to_numeric(df_transposed.index, errors='coerce')
    df_transposed = df_transposed.dropna()

    # Convert values to float
    df_transposed = df_transposed.astype(float)

    return df_selected, df_transposed


def plot_average_bar(df, title):
    # Drop 'Country Name' column if present
    df = df.drop('Country Name', axis=1, errors='ignore')

    # Calculate average values for each year
    df_avg = df.mean(axis=0)

    # Plot a bar chart
    ax = df_avg.plot(kind='bar', figsize=(12, 6), color='skyblue')

    # Set plot properties
    ax.set_ylabel("Average Values")
    ax.set_xlabel("Year")
    ax.set_title(title)

    # Show the plot
    plt.show()

def plot_multiline(df, title):
    # Plot a multiline plot
    ax = df.set_index('Country Name').T.plot(figsize=(12, 6), colormap='viridis')

    # Set plot properties
    ax.set_ylabel("Values")
    ax.set_xlabel("Year")
    ax.set_title(title)

    # Show the plot with legend using country names
    plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1, 1))

    # Show the plot
    plt.show()



def plot_correlation_heatmap(df1, df2, title):
    # Extract relevant columns (Country Name and the indicator columns)
    df_corr = pd.concat([df1['Country Name'], df1.iloc[:, 1], df2.iloc[:, 1]], axis=1)
    df_corr.columns = ['Country Name', 'CO2 emissions', 'Forest area']

    # Calculate correlation matrix
    corr_matrix = df_corr.iloc[:, 1:].corr()

    # Plot correlation heatmap using matshow
    plt.figure(figsize=(8, 6))
    plt.matshow(corr_matrix, cmap='coolwarm', fignum=1)

    # Set x and y axis labels
    plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
    plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)

    # Add colorbar
    plt.colorbar(label='Correlation')

    # Set plot properties
    plt.title(title)
    plt.show()

def scatter_plots_for_countries(df1, df2):
    # Get unique country names
    countries = df1['Country Name'].unique()

    # Calculate the number of rows and columns for subplots
    num_countries = len(countries)
    num_cols = 5  # 5 subplots for each country
    num_rows = 1

    # Create subplots without sharing axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))
    fig.suptitle("Scatter Plots of CO2 Emissions vs. Forest Area for Each Country Over Years")

    # Plot scatter plots for each country
    for i, country in enumerate(countries):
        ax = axes[i]
        country_data1 = df1[df1['Country Name'] == country].iloc[:, 1:]
        country_data2 = df2[df2['Country Name'] == country].iloc[:, 1:]

        # Flatten the data
        x_values = country_data2.values.flatten()
        y_values = country_data1.values.flatten()

        # Plot scatter plot for each country
        ax.scatter(x_values, y_values)
        ax.set_xlabel('Forest Area')
        ax.set_ylabel('CO2 Emissions')
        ax.set_title(country)

        # Set x and y axis limits based on min and max values
        ax.set_xlim([x_values.min(), x_values.max()])
        ax.set_ylim([y_values.min(), y_values.max()])

        # Calculate correlation coefficient
        correlation_coefficient = np.corrcoef(x_values, y_values)[0, 1]

        # Display correlation coefficient below the subplot
        ax.text(0.5, -0.2, f'Correlation: {correlation_coefficient:.2f}', ha='center', va='center',
                transform=ax.transAxes, color='red')

    # Adjust layout and show plots
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()



# Loading the data from the csv file
co2_emissions_data, co2_emissions_data_transposed = read_data('co2_emissions.csv')
forest_area_data, forest_area_data_transposed = read_data('forest_area.csv')

# Statistical properties of the data
print('Statistical properties of CO2 emissions over the years 1991-2020')
print(co2_emissions_data_transposed.describe())
print('\n\nStatistical properties of Forest Area over the years 1991-2020')
print(forest_area_data_transposed.describe())

# Bar Plots
plot_average_bar(co2_emissions_data, "Average CO2 Emissions over Years")
plot_average_bar(forest_area_data, "Average Forest Area over Years")

# Line Plots
plot_multiline(co2_emissions_data, "CO2 Emissions Over Years by Country")
plot_multiline(forest_area_data, "Forest Area Over Years by Country")

# Scatter Plots
scatter_plots_for_countries(co2_emissions_data, forest_area_data)

