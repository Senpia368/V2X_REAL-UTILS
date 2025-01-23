import matplotlib.pyplot as plt 

instances = {
    'Pedestrian': 234483, 
    'Car': 105056, 
    'TrashCan': 10810, 
    'Bus': 7126, 
    'ScooterRider': 6758, 
    'Truck': 4489, 
    'BicycleRider': 4096, 
    'Van': 1316, 
    'MotorcyleRider': 1141, 
    'Scooter': 288, 
    'ConstructionCart': 248, 
    'LongVehicle': 88
}

# Extract the keys and values from the dictionary
classes = list(instances.keys())
counts = list(instances.values())

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(classes, counts, color='skyblue')

# Add title and labels
plt.title('V2X Dataset Lidar Object Instances')
plt.xlabel('Class')
plt.ylabel('Number of Instances')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Save the figure
plt.tight_layout()
plt.savefig('v2x_instances_bar_chart.png')

# Display the plot
plt.show()