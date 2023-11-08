# Import the required in-built python libraries
import pandas as pd
import matplotlib.pyplot as plt

def multi_line_plot(data, rows, columns):
    # Unique markers
    markers = ['o', 's', '^', 'D', 'v', 'p']  
    
    # Unique colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  

    plt.figure(figsize=(8, 6))

    # Iterate through selected rows and plot each as a line
    for idx, row_idx in enumerate(rows):
        row_data = data.iloc[row_idx][columns]
        label = data.iloc[row_idx][0]
        marker = markers[idx]                   
        color = colors[idx]                     
        plt.plot(range(2000, 2022), row_data[:], marker=marker, color=color, label=str(label))

    # Adding labels and title
    plt.xlabel('Year')
    plt.ylabel('Percentage')
    plt.title('Access to Electricity (% of Population) in various countries from 2000 to 2021')

    # Adding legend
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()


def stacked_bar_chart(data, rows, columns):
    # Select the data to be plotted
    selected_data = data.iloc[rows, columns]
    selected_countries = data.iloc[rows, 0]
    years = data.columns[columns]
    
    plt.figure(figsize=(10, 6))
    
    # bottom_values specifies the current height of the bar. 
    bottom_values = [0] * len(selected_countries)
    
    # Iterate over the years and plot the stacked bar chart.
    for idx, year in enumerate(years):
        plt.bar(selected_countries, selected_data[year]-bottom_values, label=str(year), bottom=bottom_values)
        bottom_values = [selected_data[year].values[i] for i in range(len(selected_countries))]
    
    plt.xlabel('Country')
    plt.ylabel('Percentage of Population with Access to Electricity')
    plt.title('Electrification Rates Over Selected Years')
    plt.xticks(rotation=90)
    
    # Display the legend
    plt.legend(title='Year')
    
    # Display the plot
    plt.show()


def heatmap_plot(data, rows, columns):
    
    # Select the required data
    selected_data = data.iloc[rows, columns]
    
    # Fix the plot size
    plt.figure(figsize=(8, 6))
    
    # Set the color code
    plt.matshow(selected_data, cmap='YlGnBu')
    plt.colorbar()
    plt.xticks(range(len(columns)), data.columns[columns], rotation=90)
    plt.yticks(range(len(rows)), data.iloc[rows, 0])
    plt.xlabel('Year')
    plt.ylabel('Country')
    plt.title('Electrification Rates Heatmap')
    
    #Display the plot
    plt.show()    



# Read the dataset which is in CSV format. 
# The first 4 rows of the dataset is empty. So they are skipped. 
electricity_data = pd.read_csv("electricity.csv", skiprows=4)

# Select the index of the countries, whose data to be plotted.
countries = [1, 2, 4, 7, 82, 83]

# Select the years whose data to be plotted.
years = range(44, 66)
print(electricity_data.iloc[countries, years].head())

multi_line_plot(electricity_data, countries, years)
stacked_bar_chart(electricity_data, countries, [44, 54, 64])
heatmap_plot(electricity_data, countries, years)