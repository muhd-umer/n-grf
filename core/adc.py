# core/adc.py

import torch

from utils.transform_utils import build_rotation


def add_densification_stats(
    xyz_gradient_accum: torch.Tensor,
    denom: torch.Tensor,
    visibility_filter: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate counts for points visible in the current view.

    Args:
        xyz_gradient_accum: Accumulated position gradients
        denom: Gradient count denominator
        visibility_filter: Boolean mask of visible points

    Returns:
        Updated xyz_gradient_accum and denom tensors
    """
    denom[visibility_filter] += 1

    return xyz_gradient_accum, denom


def densify_and_split(
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    features: torch.Tensor,
    grads: torch.Tensor,
    grad_threshold: float,
    scene_extent: float,
    percent_dense: float,
    N: int = 2,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Split high-gradient Gaussians for densification.

    Args:
        xyz: Gaussian positions [N, 3]
        scaling: Scaling parameters [N, 3]
        rotation: Rotation quaternions [N, 4]
        opacity: Opacity values [N, 1]
        features: Wireless features [N, C]
        grads: Gradient magnitudes for each Gaussian [N, 1]
        grad_threshold: Threshold for selecting high-gradient Gaussians
        scene_extent: Size of the scene for scaling
        percent_dense: Minimum scale percentage for splitting
        N: Number of new Gaussians per split

    Returns:
        Tuple of new Gaussians and prune mask
    """
    n_points = xyz.shape[0]

    padded_grad = torch.zeros(n_points, device=xyz.device)
    padded_grad[: grads.shape[0]] = grads.squeeze()
    selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

    selected_pts_mask = torch.logical_and(
        selected_pts_mask,
        torch.max(scaling, dim=1).values > percent_dense * scene_extent,
    )

    if not selected_pts_mask.any():
        return None, None, None, None, None, None, 0

    stds = scaling[selected_pts_mask].repeat(N, 1)
    means = torch.zeros((stds.size(0), 3), device=xyz.device)
    samples = torch.normal(mean=means, std=stds)
    rots = build_rotation(rotation[selected_pts_mask]).repeat(N, 1, 1)

    new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz[
        selected_pts_mask
    ].repeat(N, 1)

    new_scaling = scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)

    new_rotation = rotation[selected_pts_mask].repeat(N, 1)
    new_opacity = opacity[selected_pts_mask].repeat(N, 1)
    new_features = features[selected_pts_mask].repeat(N, 1)

    prune_filter = selected_pts_mask

    return (
        new_xyz,
        new_scaling,
        new_rotation,
        new_opacity,
        new_features,
        prune_filter,
        len(new_xyz),
    )


def densify_and_clone(
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    features: torch.Tensor,
    grads: torch.Tensor,
    grad_threshold: float,
    scene_extent: float,
    percent_dense: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Clone high-gradient Gaussians in sparse regions.

    Args:
        xyz: Gaussian positions [N, 3]
        scaling: Scaling parameters [N, 3]
        rotation: Rotation quaternions [N, 4]
        opacity: Opacity values [N, 1]
        features: Wireless features [N, C]
        grads: Gradient magnitudes for each Gaussian [N, 1]
        grad_threshold: Threshold for selecting high-gradient Gaussians
        scene_extent: Size of the scene for scaling
        percent_dense: Maximum scale percentage for cloning

    Returns:
        Tuple of cloned Gaussians
    """
    selected_pts_mask = torch.where(
        torch.norm(grads, dim=-1) >= grad_threshold, True, False
    )

    selected_pts_mask = torch.logical_and(
        selected_pts_mask,
        torch.max(scaling, dim=1).values <= percent_dense * scene_extent,
    )

    if not selected_pts_mask.any():
        return None, None, None, None, None, 0

    new_xyz = xyz[selected_pts_mask]
    new_scaling = scaling[selected_pts_mask]
    new_rotation = rotation[selected_pts_mask]
    new_opacity = opacity[selected_pts_mask]
    new_features = features[selected_pts_mask]

    return new_xyz, new_scaling, new_rotation, new_opacity, new_features, len(new_xyz)


def create_prune_mask(
    opacity: torch.Tensor,
    scaling: torch.Tensor,
    max_radii2D: torch.Tensor,
    min_opacity: float,
    max_screen_size: float = None,
    extent: float = None,
) -> torch.Tensor:
    """Create a pruning mask for small/invisible or too large Gaussians.

    Args:
        opacity: Opacity values [N, 1]
        scaling: Scaling parameters [N, 3]
        max_radii2D: Maximum 2D radii for each Gaussian [N]
        min_opacity: Minimum opacity threshold
        max_screen_size: Maximum allowed screen-space size
        extent: Scene extent for scale-based pruning

    Returns:
        Boolean mask of Gaussians to prune
    """
    prune_mask = (opacity < min_opacity).squeeze()

    if max_screen_size is not None and extent is not None:
        big_points_vs = max_radii2D > max_screen_size
        big_points_ws = scaling.max(dim=1).values > 0.1 * extent

        prune_mask = torch.logical_or(
            torch.logical_or(prune_mask, big_points_vs), big_points_ws
        )

    return prune_mask


def adaptive_density_control(
    xyz_gradient_accum: torch.Tensor,
    denom: torch.Tensor,
    xyz: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    features: torch.Tensor,
    max_radii2D: torch.Tensor,
    max_grad: float,
    min_opacity: float,
    extent: float,
    max_screen_size: float = None,
    percent_dense: float = 0.01,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Apply adaptive density control to optimize Gaussian distribution.

    In this simplified version, we'll use random values for densification
    instead of gradient-based selection since we're not tracking viewspace gradients.

    Args:
        xyz_gradient_accum: Accumulated position gradients
        denom: Gradient count denominator
        xyz: Gaussian positions [N, 3]
        scaling: Scaling parameters [N, 3]
        rotation: Rotation quaternions [N, 4]
        opacity: Opacity values [N, 1]
        features: Wireless features [N, C]
        max_radii2D: Maximum 2D radii for each Gaussian [N]
        max_grad: Gradient threshold for densification
        min_opacity: Minimum opacity for pruning
        extent: Scene extent for scaling reference
        max_screen_size: Maximum allowed screen-space size
        percent_dense: Density control parameter

    Returns:
        Updated tensors (xyz, scaling, rotation, opacity, features, xyz_gradient_accum, denom)
    """
    initial_points = xyz.shape[0]
    device = xyz.device

    num_points = xyz.shape[0]
    num_to_select = max(int(0.02 * num_points), 1)

    small_gaussians = torch.max(scaling, dim=1).values <= percent_dense * extent
    small_indices = torch.where(small_gaussians)[0]
    if len(small_indices) > 0:
        perm = torch.randperm(len(small_indices), device=device)
        clone_indices = small_indices[perm[: min(num_to_select, len(small_indices))]]

        new_xyz_clone = xyz[clone_indices]
        new_scaling_clone = scaling[clone_indices]
        new_rotation_clone = rotation[clone_indices]
        new_opacity_clone = opacity[clone_indices]
        new_features_clone = features[clone_indices]
    else:
        new_xyz_clone = torch.zeros((0, 3), device=device)
        new_scaling_clone = torch.zeros((0, 3), device=device)
        new_rotation_clone = torch.zeros((0, 4), device=device)
        new_opacity_clone = torch.zeros((0, 1), device=device)
        new_features_clone = torch.zeros((0, features.shape[1]), device=device)

    large_gaussians = torch.max(scaling, dim=1).values > percent_dense * extent
    large_indices = torch.where(large_gaussians)[0]
    N = 2

    if len(large_indices) > 0:
        perm = torch.randperm(len(large_indices), device=device)
        split_indices = large_indices[perm[: min(num_to_select, len(large_indices))]]

        stds = scaling[split_indices].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(rotation[split_indices]).repeat(N, 1, 1)

        new_xyz_split = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz[
            split_indices
        ].repeat(N, 1)

        new_scaling_split = scaling[split_indices].repeat(N, 1) / (0.8 * N)

        new_rotation_split = rotation[split_indices].repeat(N, 1)
        new_opacity_split = opacity[split_indices].repeat(N, 1)
        new_features_split = features[split_indices].repeat(N, 1)

        prune_filter_split = torch.zeros(
            initial_points, dtype=torch.bool, device=device
        )
        prune_filter_split[split_indices] = True
    else:
        new_xyz_split = torch.zeros((0, 3), device=device)
        new_scaling_split = torch.zeros((0, 3), device=device)
        new_rotation_split = torch.zeros((0, 4), device=device)
        new_opacity_split = torch.zeros((0, 1), device=device)
        new_features_split = torch.zeros((0, features.shape[1]), device=device)
        prune_filter_split = torch.zeros(
            initial_points, dtype=torch.bool, device=device
        )

    prune_mask_opacity = (opacity < min_opacity).squeeze()

    if max_screen_size is not None:
        big_points_vs = max_radii2D > max_screen_size

        big_points_ws = scaling.max(dim=1).values > 0.1 * extent

        prune_mask_opacity = torch.logical_or(
            torch.logical_or(prune_mask_opacity, big_points_vs), big_points_ws
        )

    prune_mask = torch.logical_or(prune_filter_split, prune_mask_opacity)

    new_xyz_list = []
    new_scaling_list = []
    new_rotation_list = []
    new_opacity_list = []
    new_features_list = []

    if new_xyz_clone.shape[0] > 0:
        new_xyz_list.append(new_xyz_clone)
        new_scaling_list.append(new_scaling_clone)
        new_rotation_list.append(new_rotation_clone)
        new_opacity_list.append(new_opacity_clone)
        new_features_list.append(new_features_clone)

    if new_xyz_split.shape[0] > 0:
        new_xyz_list.append(new_xyz_split)
        new_scaling_list.append(new_scaling_split)
        new_rotation_list.append(new_rotation_split)
        new_opacity_list.append(new_opacity_split)
        new_features_list.append(new_features_split)

    if new_xyz_list:
        new_xyz = torch.cat(new_xyz_list, dim=0)
        new_scaling = torch.cat(new_scaling_list, dim=0)
        new_rotation = torch.cat(new_rotation_list, dim=0)
        new_opacity = torch.cat(new_opacity_list, dim=0)
        new_features = torch.cat(new_features_list, dim=0)

        xyz = torch.cat([xyz, new_xyz], dim=0)
        scaling = torch.cat([scaling, new_scaling], dim=0)
        rotation = torch.cat([rotation, new_rotation], dim=0)
        opacity = torch.cat([opacity, new_opacity], dim=0)
        features = torch.cat([features, new_features], dim=0)

        xyz_gradient_accum_new = torch.zeros((new_xyz.shape[0], 1), device=device)
        denom_new = torch.zeros((new_xyz.shape[0], 1), device=device)
        max_radii2D_new = torch.zeros(new_xyz.shape[0], device=device)

        xyz_gradient_accum = torch.cat(
            [xyz_gradient_accum, xyz_gradient_accum_new], dim=0
        )
        denom = torch.cat([denom, denom_new], dim=0)
        max_radii2D = torch.cat([max_radii2D, max_radii2D_new], dim=0)

    if prune_mask.any():
        keep_mask = ~prune_mask

        xyz = xyz[keep_mask]
        scaling = scaling[keep_mask]
        rotation = rotation[keep_mask]
        opacity = opacity[keep_mask]
        features = features[keep_mask]
        xyz_gradient_accum = xyz_gradient_accum[keep_mask]
        denom = denom[keep_mask]
        max_radii2D = max_radii2D[keep_mask]

    return (
        xyz,
        scaling,
        rotation,
        opacity,
        features,
        xyz_gradient_accum,
        denom,
        max_radii2D,
    )
