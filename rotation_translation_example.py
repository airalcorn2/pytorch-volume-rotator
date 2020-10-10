import matplotlib.animation as animation
import sys

from example_utils import *


def init_fig():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot = ax.voxels(r, g, b, init_feat_vol, facecolors=init_colors)
    ax.set(xlabel="r", ylabel="g", zlabel="b")
    return (fig, ax)


def update_plot(frame):
    transform_params = {transform_param: frame / total_frames * interval}
    new_colors = transform_func(
        init_colors, coord_map, inv_coord_map, **transform_params
    )
    new_feat_vol = new_colors[..., 3] > ALPHA_THRESH
    new_colors[new_feat_vol, 3] = 1.0
    ax.collections.clear()
    plot = ax.voxels(r, g, b, new_feat_vol, facecolors=new_colors)


if __name__ == "__main__":
    (feat_vol, colors, coord_map, inv_coord_map, vol_dim, r, g, b) = init_vol()

    # See: https://stackoverflow.com/questions/45712099/updating-z-data-on-a-surface-plot-in-matplotlib-animation.
    total_frames = 50
    if sys.argv[1] == "rotate":
        interval = 2 * np.pi
        transform_func = rotate_feature_volume
        for transform_param in ["yaw", "pitch", "roll"]:
            init_colors = colors
            init_feat_vol = feat_vol
            (fig, ax) = init_fig()
            ani = animation.FuncAnimation(fig, update_plot, total_frames, interval=100)
            # See: https://matplotlib.org/3.1.1/gallery/animation/simple_anim.html.
            ani.save(f"{transform_param}.mp4")

    elif sys.argv[1] == "translate":
        denom = 5
        interval = 2 * vol_dim / denom
        transform_func = translate_feature_volume
        for transform_param in ["x", "y", "z"]:
            init_colors = translate_feature_volume(
                colors, coord_map, inv_coord_map, **{transform_param: -vol_dim / denom}
            )
            init_feat_vol = init_colors[..., 3] > ALPHA_THRESH
            (fig, ax) = init_fig()
            ani = animation.FuncAnimation(fig, update_plot, total_frames, interval=100)
            ani.save(f"{transform_param}.mp4")
