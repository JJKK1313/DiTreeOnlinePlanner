import numpy as np
import matplotlib.pyplot as plt

maze_path = 'maps/mazes/boxes.csv'

maze_data_original = np.loadtxt(maze_path, delimiter=',')
start1 = [3, 3]
goal1 = [18, 15]

start2 = [2,17]
goal2 = [17,2]


plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)

plt.subplot(221)
obstacle_loc_row_col = [10, 15]
obstacle_dims = [1, 4]
maze_data_with_obstacle = maze_data_original.copy()
y_min = max(obstacle_loc_row_col[0], 0)
y_max = min(obstacle_loc_row_col[0] + obstacle_dims[0], maze_data_original.shape[0])
x_min = max(obstacle_loc_row_col[1], 0)
x_max = min(obstacle_loc_row_col[1] + obstacle_dims[1], maze_data_original.shape[1])
maze_data_with_obstacle[y_min:y_max, x_min:x_max] = 1

plt.imshow(2 * maze_data_with_obstacle - maze_data_original,
           origin='lower', extent=[0, maze_data_original.shape[0], 0, maze_data_original.shape[1]])
plt.scatter([start1[0]], [start1[1]], c="green", marker="x", label="Start")
plt.scatter([goal1[0]], [goal1[1]], c="red", marker="o", facecolors="none", label="Goal")
plt.legend()
plt.title("Scenario 1: 1 by 4", fontsize=20)
plt.xticks([])
plt.yticks([])

plt.subplot(222)
obstacle_loc_row_col = [9, 16]
obstacle_dims = [2, 2]
maze_data_with_obstacle = maze_data_original.copy()
y_min = max(obstacle_loc_row_col[0], 0)
y_max = min(obstacle_loc_row_col[0] + obstacle_dims[0], maze_data_original.shape[0])
x_min = max(obstacle_loc_row_col[1], 0)
x_max = min(obstacle_loc_row_col[1] + obstacle_dims[1], maze_data_original.shape[1])
maze_data_with_obstacle[y_min:y_max, x_min:x_max] = 1

plt.imshow(2 * maze_data_with_obstacle - maze_data_original,
           origin='lower', extent=[0, maze_data_original.shape[0], 0, maze_data_original.shape[1]])
plt.scatter([start1[0]], [start1[1]], c="green", marker="x", label="Start")
plt.scatter([goal1[0]], [goal1[1]], c="red", marker="o", facecolors="none", label="Goal")
plt.legend()
plt.title("Scenario 2: 2 by 2", fontsize=20)
plt.xticks([])
plt.yticks([])

plt.subplot(223)
obstacle_loc_row_col = [7, 16]
obstacle_dims = [2, 3]
maze_data_with_obstacle = maze_data_original.copy()
y_min = max(obstacle_loc_row_col[0], 0)
y_max = min(obstacle_loc_row_col[0] + obstacle_dims[0], maze_data_original.shape[0])
x_min = max(obstacle_loc_row_col[1], 0)
x_max = min(obstacle_loc_row_col[1] + obstacle_dims[1], maze_data_original.shape[1])
maze_data_with_obstacle[y_min:y_max, x_min:x_max] = 1

plt.imshow(2 * maze_data_with_obstacle - maze_data_original,
           origin='lower', extent=[0, maze_data_original.shape[0], 0, maze_data_original.shape[1]])
plt.scatter([start1[0]], [start1[1]], c="green", marker="x", label="Start")
plt.scatter([goal1[0]], [goal1[1]], c="red", marker="o", facecolors="none", label="Goal")
plt.legend()
plt.title("Scenario 3: 2 by 3", fontsize=20)
plt.xticks([])
plt.yticks([])


plt.subplot(224)
obstacle_loc_row_col = [1, 7]
obstacle_dims = [4, 4]
maze_data_with_obstacle = maze_data_original.copy()
y_min = max(obstacle_loc_row_col[0], 0)
y_max = min(obstacle_loc_row_col[0] + obstacle_dims[0], maze_data_original.shape[0])
x_min = max(obstacle_loc_row_col[1], 0)
x_max = min(obstacle_loc_row_col[1] + obstacle_dims[1], maze_data_original.shape[1])
maze_data_with_obstacle[y_min:y_max, x_min:x_max] = 1

plt.imshow(2 * maze_data_with_obstacle - maze_data_original,
           origin='lower', extent=[0, maze_data_original.shape[0], 0, maze_data_original.shape[1]])
plt.scatter([start2[0]], [start2[1]], c="green", marker="x", label="Start")
plt.scatter([goal2[0]], [goal2[1]], c="red", marker="o", facecolors="none", label="Goal")
plt.legend()
plt.title("Scenario 4: 4 by 4", fontsize=20)
plt.xticks([])
plt.yticks([])

plt.show()
