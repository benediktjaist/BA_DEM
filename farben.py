import math

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print(mcolors.CSS4_COLORS["lime"])
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], mcolors.CSS4_COLORS["lime"])
plt.show()