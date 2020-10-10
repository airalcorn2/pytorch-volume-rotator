import matplotlib

matplotlib.use("agg")

from example_utils import *
from torch import nn, optim


if __name__ == "__main__":
    (feat_vol, colors, coord_map, inv_coord_map, vol_dim, r, g, b) = init_vol()

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    in_vol_img = plot_feature_volume(feat_vol, colors, False)
    in_vol_img.save("true_in_vol.jpg")

    true_out_vols = []
    Rs = []
    Ts = []
    for (axis, rads) in [
        ("yaw", -np.pi / 3),
        ("pitch", -np.pi / 4),
        ("roll", -np.pi / 5),
    ]:
        angle_params = {axis: rads}
        new_colors = rotate_feature_volume(
            colors, coord_map, inv_coord_map, **angle_params
        )
        new_feat_vol = new_colors[..., 3] > ALPHA_THRESH

        out_vol_img = plot_feature_volume(new_feat_vol, new_colors, False)
        out_vol_img.save(f"true_{axis}_vol.jpg")

        true_out_vols.append(new_colors.transpose(3, *coord_map))
        Rs.append(gen_rotation_matrix(**angle_params))
        Ts.append(np.zeros(3)[None].T)

    true_out_vols = torch.Tensor(true_out_vols).to(device)
    Rs = torch.Tensor(Rs).to(device)
    Ts = torch.Tensor(Ts).to(device)
    in_vol = torch.rand(*true_out_vols.shape[1:], requires_grad=True, device=device)
    exp_vols = in_vol.expand((true_out_vols.shape[0], -1, -1, -1, -1))

    criterion = nn.MSELoss()
    optimizer = optim.Adam([in_vol], lr=1e-2)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    vol_imgs = []
    for epoch in range(1000):
        if (epoch % 10 == 0) and (epoch < 1000):
            vol_colors = in_vol.detach().cpu().numpy()
            vol_colors = vol_colors.transpose(
                inv_coord_map[0], inv_coord_map[1], inv_coord_map[2], 0
            )
            vol_colors[vol_colors > 1.0] = 1.0
            vol_colors[vol_colors < 0.0] = 0.0
            vol_feat_vol = vol_colors[..., 3] > ALPHA_THRESH
            vol_colors[vol_feat_vol, 3] = 1.0
            vol_img = plot_feature_volume(vol_feat_vol, vol_colors, False)
            vol_imgs.append(vol_img)
            if epoch == 0:
                vol_img.save("initial_vol.jpg")

        pred_out_vols = apply_explicit_3d(Rs, Ts, exp_vols)
        optimizer.zero_grad()
        loss = criterion(pred_out_vols, true_out_vols)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(loss.item())
            lr_scheduler.step(loss.item())

    vol_imgs[0].save(
        "optimize.gif", save_all=True, append_images=vol_imgs[1:], duration=200, loop=0
    )
