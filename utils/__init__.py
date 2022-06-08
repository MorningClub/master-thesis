from .visualization import visualize_obs, visualize_big, filter_sem, filter_palette_sem, rgb2mask
from .global_planner import RoadOption

def _numpy(x):
    return x.detach().cpu().numpy()