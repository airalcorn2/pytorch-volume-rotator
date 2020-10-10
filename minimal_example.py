from example_utils import *


if __name__ == "__main__":
    (feat_vol, colors, coord_map, inv_coord_map, vol_dim, r, g, b) = init_vol()
    plot_feature_volume(feat_vol, colors)

    angle_params = {"yaw": -np.pi / 4}
    new_colors = rotate_feature_volume(colors, coord_map, inv_coord_map, **angle_params)
    new_feat_vol = new_colors[..., 3] > ALPHA_THRESH
    new_colors[new_feat_vol, 3] = 1.0
    plot_feature_volume(new_feat_vol, new_colors)

    # Translation parameters must be scaled to match volume grid scale.
    trans_val = 0.1
    actual_width = 1.0
    scaled_trans_val = trans_val / actual_width * vol_dim
    trans_params = {"x": scaled_trans_val}
    new_colors = translate_feature_volume(
        colors, coord_map, inv_coord_map, **trans_params
    )
    new_feat_vol = new_colors[..., 3] > ALPHA_THRESH
    new_colors[new_feat_vol, 3] = 1.0
    plot_feature_volume(new_feat_vol, new_colors)
