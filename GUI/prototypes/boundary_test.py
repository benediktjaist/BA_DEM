import sympy as smp
from Boundary import Boundary
import matplotlib.pyplot as plt
import numpy as np

# (x, y)

# define the boundary
boundary = Boundary((3, 3.5), (3, 6))


# define the circle
radius = 1
center = (1, 1)


# check if boundary can be written as linear equation
if boundary.start_point[0] == boundary.end_point[0]:

    vertical_line = boundary.start_point[0]
    plt.axvline(x=vertical_line, color='r')

    # compute the distance between the center of the circle and the vertical line
    dist = abs(center[0] - boundary.start_point[0])

    # check if the circle intersects the vertical line
    if dist <= radius:
        # solve for intersection points
        y_intercepts = []
        ip1 = np.sqrt(radius ** 2 - (vertical_line - center[0]) ** 2) + center[1]
        ip2 = - np.sqrt(radius ** 2 - (vertical_line - center[0]) ** 2) + center[1]
        y_intercepts.append(ip1)
        y_intercepts.append(ip2)

        # compute the point of contact (poc)
        y_mid = sum(y_intercepts) / 2
        poc = (vertical_line, y_mid)
        print('numpy: ',poc)

    else:
        print('they do not intersect')
    # plot the circle
    x = np.linspace(center[0] - radius, center[0] + radius, 100)
    y1 = np.sqrt(radius ** 2 - (x - center[0]) ** 2) + center[1]
    y2 = -np.sqrt(radius ** 2 - (x - center[0]) ** 2) + center[1]

    plt.plot(x, y1, color='g')
    plt.plot(x, y2, color='b')

    # set the aspect ratio to 'equal' and display the plot
    plt.axis('equal')
    plt.grid(True)
    plt.show()

else:

    # compute the circle equations
    x = smp.symbols('x')
    y1 = smp.sqrt(radius ** 2 - (x - center[0]) ** 2) + center[1]
    y2 = -smp.sqrt(radius ** 2 - (x - center[0]) ** 2) + center[1]
    y3 = boundary.get_lin_eq()

    solution1 = smp.solveset(y1 - y3, x, domain=smp.Reals)
    solution2 = smp.solveset(y2 - y3, x, domain=smp.Reals)

    if solution1 or solution2:
        print(y1)
        print(y3)
        # solve for intersection points
        ip1 = smp.solve(y1 - y3, x)
        ip2 = smp.solve(y2 - y3, x)
        nullstellen = ip1 + ip2
        print(nullstellen)
        # delete duplicates
        nullstellen = [x for i, x in enumerate(nullstellen) if x not in nullstellen[:i]]

        # compute point of contact (poc)
        x_mid = sum(nullstellen) / 2
        poc = (x_mid, y3.evalf(subs={x: x_mid}))
        print('sympy: ', poc)
        diagram = smp.plot(y1, y2, y3, show=False)
        diagram.show()
    else:
        print("The equation has no real solution")
    diagram = smp.plot(y1, y2, y3, show=False)
    diagram.show()




