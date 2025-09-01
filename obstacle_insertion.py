import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image


def insert_obstacle_to_maze(maze_data: np.ndarray, obstacle_dims: list = [2, 1], start=None, goal=None):
    """
    this function gets a maze and plots it for the user. the user chooses the position for the obstacle.
    Args:
        start: start state for plot
        goal: goal state for plot
        maze_data: the maze matrix
        obstacle_dims: the dimensions of the obstacle.

    Returns: the maze with the new obstacle.
    """
    maze_with_obstacle = maze_data.copy()  # keep original for redrawing
    fig, _ = plt.subplots()
    im = plt.imshow(maze_data, origin='lower', extent=[0, maze_data.shape[0], 0, maze_data.shape[1]])
    if start is not None:
        plt.plot(start[1], start[0], 'go')
    if goal is not None:
        plt.plot(goal[1], goal[0], 'ro')

    def draw_obstacle_preview(x, y, draw=True):
        overlay = maze_with_obstacle.copy()  # reset to original
        if draw:
            x, y = int(x), int(y)
            y_min = max(y, 0)
            y_max = min(y + obstacle_dims[0], maze_data.shape[0])
            x_min = max(x, 0)
            x_max = min(x + obstacle_dims[1], maze_data.shape[1])
            overlay[y_min:y_max, x_min:x_max] = 1  # preview color (gray)
        im.set_data(overlay)
        fig.canvas.draw_idle()

    def on_motion(event):
        if not event.inaxes:
            draw_obstacle_preview(0, 0, draw=False)
            return
        draw_obstacle_preview(event.xdata, event.ydata)

    def on_click(event):
        if not event.inaxes:
            return
        x, y = int(event.xdata), int(event.ydata)
        print(x, y)
        y_min = max(y, 0)
        y_max = min(y + obstacle_dims[0], maze_data.shape[0])
        x_min = max(x, 0)
        x_max = min(x + obstacle_dims[1], maze_data.shape[1])
        maze_with_obstacle[y_min:y_max, x_min:x_max] = 1  # draw permanent
        draw_obstacle_preview(x, y)
        plt.close(fig)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_click)

    plt.show()
    return maze_with_obstacle


if __name__ == '__main__':

    mazes_dir = 'maps/mazes'
    maze_name = 'boxes'
    # Load the corresponding maze
    maze_path = os.path.join(mazes_dir, f'{maze_name}.csv')
    if not os.path.exists(maze_path):
        print(f"Maze file {maze_path} not found, skipping scenario.")

    maze_data = np.loadtxt(maze_path, delimiter=',')
    maze_data_with_obstacle = insert_obstacle_to_maze(maze_data, [4, 4])
    plt.figure()
    plt.subplot(211)
    plt.imshow(maze_data)
    plt.subplot(212)
    plt.imshow(maze_data_with_obstacle)
    plt.show()
