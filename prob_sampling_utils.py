import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


def forward_probability_map(maze, state, fov=np.pi + 1e-4):
    """
    Create a probability map over the maze biased toward nodes strictly in front of the car.

    Args:
        maze (np.ndarray): 20x20 binary map (0=free, 1=obstacle).
        state (array-like): [x, y, theta] car state.
        fov (float): forward field-of-view (radians, default 135°).

    Returns:
        prob_map (np.ndarray): 20x20 probability distribution (sums to 1).
    """
    rows, cols = maze.shape
    prob_map = np.zeros_like(maze, dtype=float)

    x, y, theta = state
    h = np.array([np.cos(theta), np.sin(theta)])  # heading unit vector

    for i in range(rows):
        for j in range(cols):
            if maze[i, j] == 1:  # obstacle → zero prob
                continue

            v = np.array([i - y, j - x])  # row difference, col difference
            dist = np.linalg.norm(v)
            if dist == 0:
                continue  # skip car cell

            cos_angle = np.dot(h, v) / dist
            if cos_angle > np.cos(fov / 2):
                prob_map[i, j] = 1  # uniform probability in forward cone
            else:
                prob_map[i, j] = 0  # outside FOV

    # Normalize to sum to 1
    total = np.sum(prob_map)
    if total > 0:
        prob_map /= total

    return prob_map


def gaussian_map(robot, goal, size=(20, 20)):
    """
    Create a discrete 2D Gaussian PDF over a map, elongated along the robot→goal line.
    Mean moves between goal (near) and midpoint (far).
    Wider sigma to cover more pixels.
    """
    H, W = size
    rx, ry = robot
    gx, gy = goal

    # Distance and direction
    dx, dy = gx - rx, gy - ry
    d = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    u = np.array([dx, dy]) / d if d > 1e-6 else np.array([1.0, 0.0])
    v = np.array([-u[1], u[0]])

    # Mean: blend goal and midpoint
    midpoint = np.array([(rx + gx) / 2, (ry + gy) / 2])
    w = -np.exp(-d / 15) + 1
    mean = (1 - w) * np.array([gx, gy]) + w * midpoint

    # Sigmas: larger spread
    sigma_long = 1.0 + 0.7 * np.log1p(d)  # main axis: starts at 1, grows faster
    sigma_side = 0.7 * sigma_long  # side axis: narrower but still noticeable

    # Covariance matrix
    R = np.stack([u, v], axis=1)
    S = np.diag([sigma_long ** 2, sigma_side ** 2])
    Sigma = R @ S @ R.T
    Sigma_inv = np.linalg.inv(Sigma)

    # Grid of points
    ys, xs = np.mgrid[0:H, 0:W]
    points = np.stack([xs, ys], axis=-1)  # shape (H, W, 2)
    diff = points - mean  # shape (H, W, 2)

    # Mahalanobis distance using @ and sum
    temp = diff @ Sigma_inv  # (H, W, 2)
    expo = np.sum(temp * diff, axis=2)  # (H, W)

    pdf = np.exp(-0.5 * expo)
    pdf[int(ry), int(rx)] = 0
    pdf /= pdf.sum()

    return pdf, mean, Sigma


def combine_multiply(prior, gauss, obstacle_mask=None, eps=1e-12):
    """
    Posterior ∝ prior * gauss
    prior: 2D array, >=0, not necessarily normalized
    gauss: 2D array, >=0, not necessarily normalized
    obstacle_mask: boolean array True=free, False=obstacle (optional)
    Returns: normalized 2D pdf (sums to 1), zeros at obstacle cells if mask provided
    """
    assert prior.shape == gauss.shape
    post = prior * gauss
    if obstacle_mask is not None:
        post = np.where(obstacle_mask, post, 0.0)
    s = post.sum()
    if s <= eps:
        # fallback: if everything zero, use prior or gauss fallback
        post = prior.copy()
        if obstacle_mask is not None:
            post = np.where(obstacle_mask, post, 0.0)
        s = post.sum()
        if s <= eps:
            # last fallback: uniform over free cells
            post = np.where(obstacle_mask if obstacle_mask is not None else np.ones_like(post, dtype=bool), 1.0, 0.0)
            s = post.sum()
    return post / s


def combine_weighted_add(prior, gauss, alpha=0.7, obstacle_mask=None, eps=1e-12):
    """
    Weighted linear blend: result = alpha*gauss + (1-alpha)*prior
    alpha in [0,1] (larger -> favors gaussian more)
    """
    assert 0.0 <= alpha <= 1.0
    assert prior.shape == gauss.shape
    post = alpha * gauss + (1.0 - alpha) * prior
    if obstacle_mask is not None:
        post = np.where(obstacle_mask, post, 0.0)
    s = post.sum()
    if s <= eps:
        # fallback as above
        post = prior.copy()
        if obstacle_mask is not None:
            post = np.where(obstacle_mask, post, 0.0)
        s = post.sum()
        if s <= eps:
            post = np.where(obstacle_mask if obstacle_mask is not None else np.ones_like(post, dtype=bool), 1.0, 0.0)
            s = post.sum()
    return post / s


def combine_log_blend(prior, gauss, beta=0.8, obstacle_mask=None, eps=1e-12):
    """
    Geometric / log blending:
      posterior ∝ exp( beta * log(prior+eps) + (1-beta) * log(gauss+eps) )
    Useful when you want geometric mean behaviour and to avoid underflow.
    """
    logp = beta * np.log(prior + eps) + (1.0 - beta) * np.log(gauss + eps)
    post = np.exp(logp) * (prior > 0)
    if obstacle_mask is not None:
        post = np.where(obstacle_mask, post, 0.0)
    s = post.sum()
    if s <= eps:
        post = prior.copy()
        if obstacle_mask is not None:
            post = np.where(obstacle_mask, post, 0.0)
        s = post.sum()
        if s <= eps:
            post = np.where(obstacle_mask if obstacle_mask is not None else np.ones_like(post, dtype=bool), 1.0, 0.0)
            s = post.sum()
    return post / s


# Sampling function
def sample_from_pdf(pdf, n_samples=30):
    flat = pdf.ravel()
    idx = np.random.choice(flat.size, size=n_samples, p=flat)
    ys, xs = np.unravel_index(idx, pdf.shape)
    return xs, ys


def main():
    maze_path = 'maps/mazes/boxes.csv'
    maze_data_original = np.loadtxt(maze_path, delimiter=',')
    start_time = time.time()
    prob_map = distance_transform_edt(1 - maze_data_original)

    robot = (3, 3)
    goal = (18, 15)
    pdf, mean, Sigma = gaussian_map(robot, goal)
    prob_map_angle = forward_probability_map(maze_data_original, state=[robot[0], robot[1], 2 * np.pi])

    # plt.imshow(prob_map_angle, origin="lower", cmap="viridis")
    # plt.colorbar(label="Probability")
    # plt.scatter([robot[0]], [robot[1]], c="red", marker="x", label="Robot")
    # plt.scatter([goal[0]], [goal[1]], c="white", marker="o", facecolors="none", label="Goal")
    # plt.scatter([mean[0]], [mean[1]], c="yellow", marker=".", label="Mean")
    # plt.legend()
    # plt.title("Gaussian Oriented Along Robot→Goal (Wider)")
    # plt.show()
    combined = combine_log_blend(prob_map, pdf, beta=0.8)
    print(f"Took: {time.time() - start_time:.5f} [sec]")
    print("Mean:", mean)
    print("Covariance matrix:\n", Sigma)

    # Sample points
    xs, ys = sample_from_pdf(combined, n_samples=15)

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    plt.subplot(221)
    hm = plt.imshow(prob_map, origin="lower", cmap="viridis")
    cax = axs[0][0].inset_axes((1.05, 0, 0.08, 1.0))
    fig.colorbar(hm, cax=cax)
    plt.scatter([robot[0]], [robot[1]], c="red", marker="x", label="Robot")
    plt.scatter([goal[0]], [goal[1]], c="green", marker="o", facecolors="none", label="Goal")
    plt.xticks([])
    plt.yticks([])
    plt.title("EDT Map", fontsize=24)

    plt.subplot(222)
    hm = plt.imshow(pdf, origin="lower", cmap="viridis")
    cax = axs[0][1].inset_axes((1.05, 0, 0.08, 1.0))
    fig.colorbar(hm, cax=cax)
    plt.scatter([robot[0]], [robot[1]], c="red", marker="x", label="Robot")
    plt.scatter([goal[0]], [goal[1]], c="green", marker="o", facecolors="none", label="Goal")
    plt.scatter([mean[0]], [mean[1]], c="magenta", marker=".", label="Mean")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.title("Gaussian Oriented\nAlong Robot → Goal", fontsize=24)

    plt.subplot(223)
    combined = combine_log_blend(prob_map, pdf, beta=0.8)
    hm = plt.imshow(combined, origin="lower", cmap="viridis")
    cax = axs[1, 0].inset_axes((1.05, 0, 0.08, 1.0))
    fig.colorbar(hm, cax=cax)
    plt.scatter([robot[0]], [robot[1]], c="red", marker="x", label="Robot")
    plt.scatter([goal[0]], [goal[1]], c="green", marker="o", facecolors="none", label="Goal")
    plt.scatter([mean[0]], [mean[1]], c="magenta", marker=".", label="Mean")
    plt.scatter(xs, ys, c="pink", s=15, label="Samples")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.title("Combined (beta=0.8)", fontsize=24)

    plt.subplot(224)
    combined = combine_log_blend(prob_map, pdf, beta=0.3)
    hm = plt.imshow(combined, origin="lower", cmap="viridis")
    cax = axs[1, 1].inset_axes((1.05, 0, 0.08, 1.0))
    fig.colorbar(hm, cax=cax)
    plt.scatter([robot[0]], [robot[1]], c="red", marker="x", label="Robot")
    plt.scatter([goal[0]], [goal[1]], c="green", marker="o", facecolors="none", label="Goal")
    plt.scatter([mean[0]], [mean[1]], c="magenta", marker=".", label="Mean")
    plt.scatter(xs, ys, c="pink", s=15, label="Samples")
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.title("Combined (beta=0.3)", fontsize=24)

    # plt.tight_layout()

    plt.savefig("Images/combined_03_08.png")
    # combined_ang = combine_multiply(combined, prob_map_angle)
    # # combined_ang = combined_ang / combined_ang.sum()
    # # Sample points
    # xs, ys = sample_from_pdf(combined_ang, n_samples=15)
    # plt.subplot(133)
    # plt.imshow(combined_ang, origin="lower", cmap="viridis")
    # plt.colorbar(label="Probability")
    # plt.scatter([robot[0]], [robot[1]], c="red", marker="x", label="Robot")
    # plt.scatter([goal[0]], [goal[1]], c="white", marker="o", facecolors="none", label="Goal")
    # plt.scatter([mean[0]], [mean[1]], c="yellow", marker=".", label="Mean")
    # plt.scatter(xs, ys, c="pink", s=15, label="Samples")
    # plt.legend()
    # plt.title("Combined")

    plt.show()


if __name__ == "__main__":
    main()
