import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 15, 7, 11, 9]

# Plotting the data
plt.figure(figsize=(8, 4))  # Optional: Set the figure size
plt.plot(x, y, marker='o', linestyle='-', color='b', label='Line')  # Plotting the line
plt.xlabel('X Axis')  # Label for the x-axis
plt.ylabel('Y Axis')  # Label for the y-axis
plt.title('Simple Line Plot')  # Title of the plot
plt.legend()  # Display legend based on the 'label' parameter in plt.plot()
plt.grid(True)  # Show grid
plt.show()  # Display the plot
