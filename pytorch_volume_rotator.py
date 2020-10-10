import torch


def apply_explicit_3d(Rs, Ts, input_volumes, use_einsum=False):
    """This function uses trilinear interpolation to rotate and translate feature volumes.

    :param Rs: a Tensor of rotation matrices with shape [batch_size, 3, 3].
    :param Ts: a Tensor of translation vectors (i.e., columns) with shape [batch_size, 3, 1].
    :param input_volumes: a batch of feature volumes with shape [batch_size, n_feats, height, width, depth].
    :param use_einsum: whether or not to use torch.einsum for interpolation. Here for pedagogical purposes.
    :return: the rotated and translated feature volume following trilinear interpolation.
    """
    # Initialize three volumes containing the x, y, and z-coordinates, respectively, of
    # each output voxel. Each volume has shape [height, width, depth].
    # Each voxel in x[:, col, :] has the same x-coordinate, each voxel in y[row, :, :]
    # has the same y-coordinate (and y[0, :, :] contains the *highest* value, i.e., it's
    # the "top"), and each voxel in z[:, :, depth] has the same z-coordinate.
    (height, width, depth) = input_volumes.shape[2:]
    xs = torch.linspace(0, width - 1, width).repeat(height, depth, 1).permute(0, 2, 1)
    ys = torch.linspace(0, height - 1, height).repeat(width, depth, 1).permute(2, 0, 1)
    ys = (height - 1) - ys
    zs = torch.linspace(0, depth - 1, depth).repeat(height, width, 1)

    # Combine and arrange the volume coordinates into a matrix with shape
    # [3, height * width * depth], i.e.,
    # xyzs[:, row * (width * depth) + col * depth + dep] contains the coordinates of the
    # output voxel at (row, col, dep).
    xyzs = torch.stack([xs.flatten(), ys.flatten(), zs.flatten()]).to(Rs.device)

    # Translate the voxel coordinates so that the center of the volume has coordinates
    # (0, 0, 0), and replicate the volume coordinates batch_size times, i.e.,
    # xyz_centereds has shape [batch_size, 3, height * width * depth].
    center_trans = (
        torch.Tensor([[width / 2], [height / 2], [depth / 2]]).to(Rs.device) - 0.5
    )
    xyzs_centered = (xyzs - center_trans).repeat(len(Rs), 1, 1)

    # To fill in the voxels of an *output* volume, we need to find the corresponding
    # coordinates in the input volume, so we *subtract* the translations from the output
    # voxel coordinates and apply the *inverse* rotations to the translated coordinates.
    xyzs_trans = xyzs_centered - Ts
    R_invs = Rs.permute(0, 2, 1)
    xyzs_input = torch.bmm(R_invs, xyzs_trans)

    # Do trilinear interpolation.
    # See: https://en.wikipedia.org/wiki/Trilinear_interpolation.
    # Note: the coordinate differences are not scaled because the distance between
    # lattice points is one.
    # Note: the Wikipedia article uses a different coordinate convention, so the z and y
    # steps are swapped.

    # For each input coordinate (x, y, z), get the indexes for the eight surrounding
    # input voxels that will be used to perform trilinear interpolation. cols, rows, and
    # deps have shape [batch_size, height * width * depth, 8].
    xyz_max = torch.Tensor([[width - 1], [height - 1], [depth - 1]]).to(R_invs.device)
    zeros = torch.zeros((3, 1)).to(R_invs.device)
    xyzs_uncentered = xyzs_input + center_trans
    xyzs_clipped = torch.max(zeros, torch.min(xyz_max, xyzs_uncentered))
    cols = (
        torch.cat(
            [
                torch.floor(xyzs_clipped[:, 0])[:, None, :].repeat(1, 4, 1),
                torch.ceil(xyzs_clipped[:, 0])[:, None, :].repeat(1, 4, 1),
            ],
            dim=1,
        )
        .permute(0, 2, 1)
        .long()
    )
    rows = (
        height
        - 1
        - torch.cat(
            [
                torch.floor(xyzs_clipped[:, 1])[:, None, :],
                torch.ceil(xyzs_clipped[:, 1])[:, None, :],
            ],
            dim=1,
        )
        .repeat(1, 4, 1)
        .permute(0, 2, 1)
        .long()
    )
    deps = (
        torch.cat(
            [
                torch.floor(xyzs_clipped[:, 2])[:, None, :].repeat(1, 2, 1),
                torch.ceil(xyzs_clipped[:, 2])[:, None, :].repeat(1, 2, 1),
            ],
            dim=1,
        )
        .repeat(1, 2, 1)
        .permute(0, 2, 1)
        .long()
    )

    # Builds a Tensor containing the features of the voxels that will be used to perform
    # trilinear interpolation. cxxxs has shape
    # [batch_size, height * width * depth, 8, n_feats].
    perm_volume = input_volumes.permute(0, 2, 3, 4, 1)
    cxxxs = []
    for i in range(len(input_volumes)):
        cxxxs.append(perm_volume[i, rows[i], cols[i], deps[i], :])

    cxxxs = torch.stack(cxxxs)

    # Calculate the differences.
    xdydzds = (xyzs_clipped - torch.floor(xyzs_clipped)).permute(0, 2, 1)

    if use_einsum:
        ein_str = "bvnf,bv->bvnf"
        cs = cxxxs
        for (rd_idx, xyz_idx) in enumerate([0, 2, 1]):
            mid = 8 // (2 ** (rd_idx + 1))
            cs_l = torch.einsum(ein_str, cs[:, :, 0:mid], (1 - xdydzds[..., xyz_idx]))
            cs_r = torch.einsum(ein_str, cs[:, :, mid : 2 * mid], xdydzds[..., xyz_idx])
            cs = cs_l + cs_r

    else:
        xdydzds = xdydzds.unsqueeze(-1)
        (xds, yds, zds) = (xdydzds[:, :, 0:1], xdydzds[:, :, 1:2], xdydzds[:, :, 2:3])
        cxxs = cxxxs[:, :, [0, 1, 2, 3]] * (1 - xds) + cxxxs[:, :, [4, 5, 6, 7]] * xds
        cxs = cxxs[:, :, [0, 1]] * (1 - zds) + cxxs[:, :, [2, 3]] * zds
        cs = cxs[:, :, [0]] * (1 - yds) + cxs[:, :, [1]] * yds

    output_volumes = cs.squeeze(2).permute(0, 2, 1).view(*input_volumes.shape)

    return output_volumes
