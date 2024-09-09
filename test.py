import matplotlib.pyplot as plt

# Create a simple plot
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Test Plot')
plt.xlabel('X axis')
plt.ylabel('Y axis')

# Try different ways to display/save the plot
plt.savefig('test_plot.png')  # Save as an image file
plt.show()  # Attempt to display the plot