import numpy as np
import matplotlib.pyplot as plt


class Lidar2DSim:
    def __init__(self, azimuth_fov_deg=360, azimuth_res_deg=2.0, max_range=300, noise_std=0.0, scan_time=0.2):
        self.azimuth_fov = azimuth_fov_deg
        self.azimuth_res = azimuth_res_deg
        self.max_range = max_range
        self.noise_std = noise_std

        self.scan_time = scan_time  # the frequency of full circle scan (1 / time to do full scan)

        self.angles_deg = np.arange(-self.azimuth_fov / 2,
                                    self.azimuth_fov / 2 + self.azimuth_res,
                                    self.azimuth_res)

    def scan(self, robot_state, maze_data, debug=False):
        all_distances = []
        all_endpoints = []
        yaw = robot_state[-1]
        visited_points = []
        for angle in self.angles_deg:
            distance, endpoint, ray_trace_points_indices = self._cast_ray(robot_state, maze_data, angle)
            visited_points.extend(ray_trace_points_indices)
            # Compute Ray Angle
            ray_angle_deg = yaw + angle
            ray_angle_rad = np.deg2rad(ray_angle_deg)
            dx, dy = np.cos(ray_angle_rad), np.sin(ray_angle_rad)
            # Add Gaussian noise to the range
            noisy_distance = distance + np.random.normal(0, self.noise_std)
            noisy_distance = np.clip(noisy_distance, 0, self.max_range)

            # Adjust the endpoint accordingly
            endpoint_noisy = (
                robot_state[0] + noisy_distance * dx,
                robot_state[1] + noisy_distance * dy
            )

            all_distances.append(noisy_distance)
            all_endpoints.append(endpoint_noisy)

        if debug:
            self.plot_scan(robot_state, maze_data, all_endpoints, all_distances)
        return np.array(all_distances), np.array(all_endpoints), np.array(visited_points)

    @staticmethod
    def _cast_ray(robot_state, maze_data, angle):
        # Extract Data
        x0, y0, yaw = robot_state
        maze_width, maze_height = maze_data.shape
        # Compute Ray Angle
        ray_angle_deg = yaw + angle
        ray_angle_rad = np.deg2rad(ray_angle_deg)
        ray_vector = np.array([np.cos(ray_angle_rad), np.sin(ray_angle_rad)])
        robot_pos = np.array([x0, y0])

        # Define map borders
        borders = [
            [np.array([0, 0]), np.array([0, maze_height])],  # Left
            [np.array([maze_width, 0]), np.array([maze_width, maze_height])],  # Right
            [np.array([0, 0]), np.array([maze_width, 0])],  # Bottom
            [np.array([0, maze_height]), np.array([maze_width, maze_height])],  # Top
        ]
        # Border direction vectors
        d = list(map(lambda x: x[1] - x[0], borders))

        # Find intersection of Ray with one of the borders.
        last_point = None
        for i in range(len(borders)):
            di = d[i].astype(float)
            A = np.column_stack((ray_vector, -di))
            b = borders[i][0] - robot_pos

            try:
                ts = np.linalg.solve(A, b)
                t, s = ts
                if t >= 0 and 1 >= s >= 0:  # Ray must go forward; segment must be on edge
                    last_point = t * ray_vector + robot_pos
                    break
            except np.linalg.LinAlgError:
                continue  # Ray and border are parallel

        # sample points along the ray
        t = np.arange(0, 1, step=0.1/np.linalg.norm(last_point-robot_pos))
        ray_dots = robot_pos[np.newaxis] + t[np.newaxis].T * (last_point - robot_pos)[np.newaxis]   #
        # quantize the points to fit into maze indices
        ray_dots_quantized = np.floor(ray_dots).astype(int)
        ray_dots_quantized = np.clip(ray_dots_quantized, [0, 0], [maze_width - 1, maze_height - 1])
        # find first occurrence of obstacle
        is_there_obstacle = maze_data[ray_dots_quantized.T[1], ray_dots_quantized.T[0]]
        # get real position of the obstacle.
        obstacle_pos = ray_dots[is_there_obstacle == 1][0] if np.any(is_there_obstacle == 1) else last_point
        first_obs_indx = np.where(is_there_obstacle == 1)[0][0] if np.any(is_there_obstacle == 1) else len(ray_dots_quantized)
        ray_dots_quantized = ray_dots_quantized[:first_obs_indx]
        dist = np.linalg.norm(obstacle_pos - robot_pos)
        # print(f"Angle {ray_angle_deg:+6.1f}°: {dist:5.2f}")
        return dist, obstacle_pos, ray_dots_quantized

    def plot_scan(self, robot_state, maze_data, endpoints, distances):
        plt.figure(figsize=(10, 10))

        # --- Top subplot: 2D Map with rays ---
        plt.imshow(maze_data, cmap='gray_r', origin='lower', extent=[0, maze_data.shape[0], 0, maze_data.shape[1]])
        endpoints = np.array(endpoints)
        for pt in endpoints:
            if np.linalg.norm(pt - robot_state[:2]) == 0:
                continue
            vec = (pt - robot_state[:2])/np.linalg.norm(pt - robot_state[:2])
            robot_state_plus_eps = vec + robot_state[:2]
            plt.plot([robot_state_plus_eps[0], pt[0]], [robot_state_plus_eps[1], pt[1]], 'r-', linewidth=0.5)
            plt.plot(pt[0], pt[1], 'go', markersize=2)
        plt.scatter(robot_state[0], robot_state[1], c='blue', label='Lidar')
        plt.quiver(robot_state[0], robot_state[1], np.cos(-robot_state[2]), np.sin(-robot_state[2]))
        plt.title("LiDAR Scan Simulation (Top-Down View)")
        plt.legend()
        plt.grid('major')
        plt.gca().set_aspect('equal')
        # # --- Bottom subplot: Polar Plot ---
        # ax_polar = plt.subplot(2, 1, 2, polar=True)
        #
        # # Convert angles to radians (relative to forward direction)
        # azimuths_rad = np.deg2rad(self.angles_deg)
        #
        # ax_polar.plot(azimuths_rad, distances, 'g.-')
        # ax_polar.set_theta_zero_location('N')  # Zero at top
        # ax_polar.set_theta_direction(-1)  # Clockwise
        # ax_polar.set_title("Measured Range vs Azimuth (Polar Plot)", va='bottom')
        # ax_polar.set_rlabel_position(135)

        plt.tight_layout()
        plt.show()


def main():
    # Create a simple 2D map: 0 = free, 1 = obstacle
    map_size = (100, 100)
    env = np.zeros(map_size, dtype=int)
    env[10:90, 30:50] = 1  # Vertical wall
    env[70, 30:70] = 1  # Horizontal wall

    # LiDAR setup
    lidar_pos = (10, 80, 90)  # x, y, yaw position
    rays_per_azimuth = 1  # number of rays per direction

    lidar = Lidar2DSim()

    measurements, endpoints, visited_points = lidar.scan(lidar_pos, env)
    endpoints_rounded = np.round(endpoints).astype('int')
    endpoints_rounded[:, 0] = np.clip(endpoints_rounded[:, 0], 0, env.shape[0] - 1)
    endpoints_rounded[:, 1] = np.clip(endpoints_rounded[:, 1], 0, env.shape[1] - 1)
    plt.subplot(211)
    plt.imshow(env, origin='lower', extent=[0, map_size[0], 0, map_size[1]])
    plt.subplot(212)
    env_sampled = np.zeros(map_size, dtype=int)
    env_sampled[endpoints_rounded[:, 1], endpoints_rounded[:, 0]] = 1
    plt.imshow(env_sampled)
    plt.show(block=False)

    lidar.plot_scan(lidar_pos, env, endpoints, measurements)

    # Print distances for each angle
    print("Measured Distances (in pixels):")
    for angle, dist in zip(lidar.angles_deg, measurements[::rays_per_azimuth]):
        print(f"Angle {angle:+6.1f}°: {dist:5.2f}")


if __name__ == '__main__':
    main()
