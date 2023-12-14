# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Read data from CSV file
data = pd.read_csv('data9.csv', header=None, names=['Salary'])
salaries = data['Salary']

print(data.describe())

# Fit a normal distribution to the data
fit_params = norm.fit(salaries)

# Generate PDF values using the fit parameters
pdf_values = norm.pdf(salaries, *fit_params)

# Plot histogram of salaries
plt.hist(salaries, bins=30, density=True, alpha=0.6, color='g', label='Histogram of Salaries')

# Plot the PDF curve using the fitted parameters
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000000)
p = norm.pdf(x, *fit_params)
plt.plot(x, p, 'k', linewidth=2, label='PDF of salaries')

# Add labels, title, and legend
plt.xlabel('Annual Salary (Euros)')
plt.ylabel('Probability Density')
plt.title('Salary Distribution')
plt.legend()

# Calculate the mean using trapezoidal rule for numerical integration
mean_salary = np.trapz(x*p, x)

# Calculate X from the PDF
X = norm.cdf(1.2 * mean_salary, *fit_params) - norm.cdf(0.8 * mean_salary, *fit_params)

# Annotate mean and X on the graph at specified coordinates
plt.annotate(f'$W$: {mean_salary:.2f} Euros', xy=(0.45, 0.7), xycoords='axes fraction')
plt.annotate(f'X: {X:.2f}', xy=(0.45, 0.6), xycoords='axes fraction')

# Show the plot
plt.show()
