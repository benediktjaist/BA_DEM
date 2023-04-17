import numpy as np
import matplotlib.pyplot as plt

def intersect_circle_line(center, radius, point1, point2):
    direction_line = point2 - point1
    direction_p1_center = center - point1 # v

    # calculate projection direction_p1_center on direction line
    projection = np.dot(direction_line, direction_p1_center)/np.dot(direction_line, direction_line) * direction_line

    distance = np.sqrt(np.linalg.norm(direction_p1_center)**2 - np.linalg.norm(projection)**2)

    interpenetration = distance - radius

    return interpenetration, projection, distance

point1 = np.array([2,1])
point2 = np.array([4,2])
center = np.array([2,2])

interpenetration, projection, distance = intersect_circle_line(center, 1, point1, point2)

print(projection)
print(distance)
# Plot the points and center
plt.scatter(point1[0], point1[1], color='green')
plt.scatter(point2[0], point2[1], color='green')
plt.scatter(center[0], center[1], color='red')

# Plot the direction_line in green
plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='green')

# Plot the direction_p1_center in blue
plt.plot([point1[0], center[0]], [point1[1], center[1]], color='blue')

# Plot the projection in red
plt.plot([point1[0], point1[0]+projection[0]], [point1[1], point1[1]+projection[1]], color='red')

plt.grid(True)
plt.axis('equal')
# Show the plot
plt.show()


