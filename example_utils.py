import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from pytorch_volume_rotator import apply_explicit_3d

ALPHA_THRESH = 0.5


def init_vol(vol_dim=16, shape="cube"):
    # See: https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_rgb.html.
    # The color channel coordinates are in a weird order when plotted in Matplotlib.
    coord_map = (2, 0, 1)
    inv_coord_map = {
        new_coord: 1 + coord for (coord, new_coord) in enumerate(coord_map)
    }
    (r, g, b) = np.indices((vol_dim + 1, vol_dim + 1, vol_dim + 1)) / vol_dim
    rc = midpoints(r)
    gc = midpoints(g)
    bc = midpoints(b)

    if shape == "sphere":
        feat_vol = (rc - 0.5) ** 2 + (gc - 0.5) ** 2 + (bc - 0.5) ** 2 < 0.35 ** 2

    elif shape == "cube":
        voxel_truths = []
        for color_c in [rc, gc, bc]:
            voxel_truths.append(np.abs(color_c - 0.5) < 0.25)

        feat_vol = np.logical_and.reduce(voxel_truths)

    colors = np.ones(feat_vol.shape + (4,))
    colors[..., :3] = 0.5
    colors[feat_vol, 0] = rc[feat_vol]
    colors[feat_vol, 1] = gc[feat_vol]
    colors[feat_vol, 2] = bc[feat_vol]
    colors[~feat_vol, 3] = 0.0

    return (feat_vol, colors, coord_map, inv_coord_map, vol_dim, r, g, b)


def gen_single_angle_rotation_matrix(which_angle, angle):
    """Generate a rotation matrix for one of the yaw, pitch, or roll angles.

    :param which_angle:
    :param angle:
    :return:
    """
    if which_angle == "yaw":
        (first_idx, second_idx) = (0, 2)
        negs = np.array([1.0, 1.0, -1.0, 1.0])
    elif which_angle == "pitch":
        (first_idx, second_idx) = (1, 2)
        negs = np.array([1.0, -1.0, 1.0, 1.0])
    elif which_angle == "roll":
        (first_idx, second_idx) = (0, 1)
        negs = np.array([1.0, -1.0, 1.0, 1.0])

    R = np.eye(3)
    R[first_idx, first_idx] = negs[0] * np.cos(angle)
    R[first_idx, second_idx] = negs[1] * np.sin(angle)
    R[second_idx, first_idx] = negs[2] * np.sin(angle)
    R[second_idx, second_idx] = negs[3] * np.cos(angle)
    return R


def gen_rotation_matrix(yaw=0.0, pitch=0.0, roll=0.0):
    """Generate a rotation matrix from yaw, pitch, and roll angles (in radians).
    :param yaw:
    :param pitch:
    :param roll:
    :return:
    """
    R_yaw = gen_single_angle_rotation_matrix("yaw", yaw)
    R_pitch = gen_single_angle_rotation_matrix("pitch", pitch)
    R_roll = gen_single_angle_rotation_matrix("roll", roll)
    return R_yaw @ R_pitch @ R_roll


def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]

    return x


def plot_feature_volume(feat_vol, colors, show=True):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    vol_dim = feat_vol.shape[0]
    (r, g, b) = np.indices((vol_dim + 1, vol_dim + 1, vol_dim + 1)) / vol_dim
    ax.voxels(r, g, b, feat_vol, facecolors=colors)
    ax.set(xlabel="r", ylabel="g", zlabel="b")

    if show:
        plt.show()
    else:
        canvas = plt.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = Image.frombytes(
            "RGB", canvas.get_width_height(), canvas.tostring_rgb()
        )
        plt.close("all")
        return pil_image


def rotate_feature_volume(
    original_colors, coord_map, inv_coord_map, yaw=0.0, pitch=0.0, roll=0.0
):
    Rs = torch.Tensor(gen_rotation_matrix(yaw, pitch, roll))[None]
    Ts = torch.zeros(3)[None].unsqueeze(2)
    input_volumes = torch.Tensor(original_colors.transpose(3, *coord_map)[None])
    output_volumes = apply_explicit_3d(Rs, Ts, input_volumes)
    output_volumes = output_volumes.detach().cpu().numpy()[0]
    return output_volumes.transpose(
        inv_coord_map[0], inv_coord_map[1], inv_coord_map[2], 0
    )


def translate_feature_volume(
    original_colors, coord_map, inv_coord_map, x=0.0, y=0.0, z=0.0
):
    Rs = torch.eye(3)[None]
    Ts = torch.Tensor([x, y, z])[None].unsqueeze(2)
    input_volumes = torch.Tensor(original_colors.transpose(3, *coord_map)[None])
    output_volumes = apply_explicit_3d(Rs, Ts, input_volumes)
    output_volumes = output_volumes.detach().cpu().numpy()[0]
    return output_volumes.transpose(
        inv_coord_map[0], inv_coord_map[1], inv_coord_map[2], 0
    )
