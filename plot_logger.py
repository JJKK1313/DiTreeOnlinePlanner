# plot_logger.py
import io
import os


class PlotLogger:
    def __init__(self):
        self.frames = []  # list of (step, bytes) or (step, fig) etc.

    def save_fig(self, fig, save_path):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)  # or "jpg"
        buf.seek(0)
        self.frames.append((save_path, buf.getvalue()))  # store raw bytes

    def flush_to_disk(self):
        # os.makedirs(out_dir, exist_ok=True)
        for i, (save_path, data) in enumerate(self.frames):
            with open(f"{save_path}", "wb") as f:
                f.write(data)


# Global singleton
plot_logger = PlotLogger()
