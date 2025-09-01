# import matplotlib.pyplot as plt
# from plot_logger import plot_logger

#
# class PlotManager:
#     _instance = None
#
#     def __new__(cls):
#         if cls._instance is None:
#             cls._instance = super(PlotManager, cls).__new__(cls)
#             cls._instance._init_figure()
#         return cls._instance
#
#     def _init_figure(self):
#         self.fig, self.ax = plt.subplots()
#
#     # --- figure management ---
#     def clear(self):
#         """Clear the current axes."""
#         self.ax.cla()
#
#     def subplot(self, nrows, ncols, index):
#         """Switch to a specific subplot."""
#         self.ax = self.fig.add_subplot(nrows, ncols, index)
#
#     # --- plotting functions ---
#     def plot(self, *args, **kwargs):
#         self.ax.plot(*args, **kwargs)
#
#     def scatter(self, *args, **kwargs):
#         self.ax.scatter(*args, **kwargs)
#
#     def imshow(self, *args, **kwargs):
#         self.ax.imshow(*args, **kwargs)
#
#     def quiver(self, *args, **kwargs):
#         """Plot a quiver plot (vectors)."""
#         self.ax.quiver(*args, **kwargs)
#
#     # --- figure metadata ---
#     def title(self, title_str, **kwargs):
#         self.ax.set_title(title_str, **kwargs)
#
#     def xlabel(self, label_str, **kwargs):
#         self.ax.set_xlabel(label_str, **kwargs)
#
#     def ylabel(self, label_str, **kwargs):
#         self.ax.set_ylabel(label_str, **kwargs)
#
#     def axis(self, *args, **kwargs):
#         self.ax.axis(*args, **kwargs)
#
#     def legend(self, *args, **kwargs):
#         """Show legend on the axes."""
#         self.ax.legend(*args, **kwargs)
#
#     def fill(self, *args, **kwargs):
#         """Wrapper for ax.fill"""
#         self.ax.fill(*args, **kwargs)
#
#     # --- saving / logging ---
#     def save(self, save_path=None, log_to_buffer=True, dpi=100):
#         """
#         Save the figure.
#         If log_to_buffer=True, saves bytes to PlotLogger buffer instead of directly to disk.
#         """
#         if save_path is None:
#             return
#         if log_to_buffer:
#             plot_logger.save_fig(self.fig, save_path)
#         else:
#             self.fig.savefig(save_path, dpi=dpi)
#
#     def show(self):
#         """Display the figure (optional for debugging)."""
#         self.fig.show()
#
#
# plot_manager = PlotManager()
