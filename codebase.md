This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: utils, models, engine/, datasets/*m, datasets/*.py, train.py, eval.py
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

## Additional Info

# Directory Structure
```
datasets/
  __init__.py
  create_users.m
  dataloader.py
  fspl.m
  generate_csi.m
  generate_pc.m
  get_ray_chan.m
  indoor.m
  outdoor.m
  ray_marching.m
  wireless_dataset.py
engine/
  __init__.py
  render_magnitude.py
models/
  __init__.py
  gaussian_model.py
  networks.py
utils/
  __init__.py
  general_utils.py
  loss.py
  pos_encoder.py
  train_utils.py
  transform_utils.py
eval.py
train.py
```

# Files

## File: datasets/__init__.py
```python
# datasets/__init__.py

from .dataloader import get_dataloaders, get_wireless_dataloader
from .wireless_dataset import WirelessDataset
```

## File: datasets/create_users.m
```
function [Users, num_created] = create_users(env_dims, num_users, rx_array, user_params, stl_data)
    % CREATE_USERS Generate user positions in a 3D environment
    %
    % Description:
    %   Creates user receiver sites within specified environmental dimensions
    %   with optional collision detection for buildings and other users.
    %
    % Inputs:
    %   env_dims    - [3x2] Matrix defining environment bounds [xmin xmax; ymin ymax; zmin zmax]
    %   num_users   - Number of users to generate
    %   rx_array    - Antenna array configuration for receivers
    %   user_params - Structure with the following optional fields:
    %                   .check_building_collision - Enable building collision (default: false)
    %                   .check_user_collision    - Enable user separation (default: true)
    %                   .separation_distance     - Minimum distance between users (auto)
    %   stl_data    - STL mesh data for building collision detection (optional)
    %
    % Outputs:
    %   Users       - Array of rxsite objects representing user positions
    %   num_created - Actual number of users created (may be less than requested)
    %
    % Example:
    %   env_dims = [-10 10; -10 10; 0 20];
    %   num_users = 10;
    %   rx_array = phased.ULA('NumElements', 4);
    %   params.check_building_collision = true;
    %   params.separation_distance = 2;
    %   [users, created] = create_users(env_dims, num_users, rx_array, params);

    % default parameters if not provided
    if ~isfield(user_params, 'check_building_collision')
        user_params.check_building_collision = false;
    end

    if ~isfield(user_params, 'check_user_collision')
        user_params.check_user_collision = true;
    end

    if ~isfield(user_params, 'separation_distance')
        % calculate default separation based on environment size and user count
        env_area = (env_dims(1, 2) - env_dims(1, 1)) * (env_dims(2, 2) - env_dims(2, 1));
        user_params.separation_distance = sqrt(env_area / (num_users * pi)) / 2;
    end

    Users(num_users) = rxsite;
    valid_users = 0;
    max_attempts = num_users * 50;
    attempts = 0;

    % store valid positions for faster collision checking
    valid_positions = zeros(3, num_users);

    % preprocess stl data if building collision check is enabled
    if user_params.check_building_collision && ~isempty(stl_data)
        vertices = stl_data.Points;
        faces = stl_data.ConnectivityList;
    end

    while valid_users < num_users && attempts < max_attempts
        pos = [ ...
                   unifrnd(env_dims(1, 1), env_dims(1, 2)); ...
                   unifrnd(env_dims(2, 1), env_dims(2, 2)); ...
                   unifrnd(0.3, 1.8) ...
               ];

        is_valid = true;

        % check building collision if enabled
        if user_params.check_building_collision && ~isempty(stl_data)
            is_valid = ~is_point_in_building(pos', vertices, faces);
        end

        % check user collision if enabled and passed building check
        if is_valid && user_params.check_user_collision && valid_users > 0

            for i = 1:valid_users

                if norm(pos(1:2) - valid_positions(1:2, i)) < user_params.separation_distance
                    is_valid = false;
                    break;
                end

            end

        end

        if is_valid
            valid_users = valid_users + 1;
            valid_positions(:, valid_users) = pos;
            Users(valid_users) = rxsite("cartesian", ...
                "Antenna", rx_array, ...
                "AntennaPosition", pos);
        end

        attempts = attempts + 1;
    end

    if valid_users < num_users
        warning('could not place all users after %d attempts', max_attempts);
        Users = Users(1:valid_users);
    end

    num_created = valid_users;
end

function inside = is_point_in_building(point, vertices, faces)
    % ray casting algorithm for point-in-mesh detection
    ray_direction = [1, 0, 0];
    intersections = 0;

    for i = 1:size(faces, 1)
        triangle = vertices(faces(i, :), :);

        if does_ray_intersect_triangle(point, ray_direction, triangle)
            intersections = intersections + 1;
        end

    end

    inside = mod(intersections, 2) == 1;
end

function intersects = does_ray_intersect_triangle(origin, direction, triangle)
    % möller-trumbore ray-triangle intersection algorithm
    epsilon = 1e-7;

    edge1 = triangle(2, :) - triangle(1, :);
    edge2 = triangle(3, :) - triangle(1, :);
    h = cross(direction, edge2);
    a = dot(edge1, h);

    if abs(a) < epsilon
        intersects = false;
        return;
    end

    f = 1 / a;
    s = origin - triangle(1, :);
    u = f * dot(s, h);

    if u < 0.0 || u > 1.0
        intersects = false;
        return;
    end

    q = cross(s, edge1);
    v = f * dot(direction, q);

    if v < 0.0 || u + v > 1.0
        intersects = false;
        return;
    end

    t = f * dot(edge2, q);
    intersects = t > epsilon;
end
```

## File: datasets/fspl.m
```
function pl = fspl(d, f)
    % FSPL Compute free-space path loss (FSPL)
    %
    % Description:
    %   Calculates the free-space path loss in dB for a given distance and frequency.
    %
    % Inputs:
    %   d - Distance between transmitter and receiver (meters)
    %   f - Frequency (Hz)
    %
    % Output:
    %   pl - Path loss in dB
    %
    % Example:
    %   pl = fspl(100, 2.4e9);

    pl = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/physconst("lightspeed"));
end
```

## File: datasets/generate_pc.m
```
function all_points = generate_pc(vertices, faces, params, env_dims, visualize)
    % GENERATE_PC Generate point clouds for a given environment
    %
    % Description:
    %   Generates a point cloud representation of an environment using multiple
    %   sampling strategies including edge, surface, volume, and boundary points.
    %   Supports optional DBSCAN clustering and visualization.
    %
    % Inputs:
    %   vertices   - [Nx3] Matrix of vertex coordinates from STL
    %   faces      - [Mx3] Matrix of face indices from STL
    %   params     - Structure with the following optional fields:
    %                  .edge_density      - Points per edge unit length (default: 0)
    %                  .surface_density   - Points per triangle unit area (default: 0)
    %                  .volume_density    - Points per unit volume (default: 0)
    %                  .boundary_density  - Points per boundary surface area (default: 0)
    %                  .random_points     - Additional random points in volume (default: 0)
    %                  .noise_std         - Standard deviation for perturbation (default: 0)
    %                  .edge_reduction    - Factor to reduce edge points (default: 1)
    %                  .surface_reduction - Factor to reduce surface points (default: 1)
    %                  .use_dbscan        - Enable DBSCAN clustering (default: false)
    %                  .dbscan_epsilon    - DBSCAN epsilon parameter (default: 0.2)
    %                  .dbscan_minpts     - DBSCAN minimum points (default: 5)
    %   env_dims   - [3x2] Matrix of environment bounds [min_x max_x; min_y max_y; min_z max_z]
    %   visualize  - (Optional) Boolean to enable visualization (default: false)
    %
    % Output:
    %   all_points - [Px3] Matrix of generated point cloud coordinates
    %
    % Example:
    %   [v, f] = stlread('building.stl');
    %   params.edge_density = 1;
    %   params.surface_density = 0.5;
    %   env_dims = [-10 10; -10 10; 0 20];
    %   points = generate_pc(v, f, params, env_dims, true);

    % set default values if not provided
    default_params = struct('edge_density', 0, ...
        'surface_density', 0, ...
        'volume_density', 0, ...
        'boundary_density', 0, ...
        'random_points', 0, ...
        'noise_std', 0, ...
        'edge_reduction', 1, ...
        'surface_reduction', 1, ...
        'use_dbscan', false, ...
        'dbscan_epsilon', 0.2, ...
        'dbscan_minpts', 5);

    % merge provided params with defaults
    if ~isfield(params, 'edge_density'), params.edge_density = default_params.edge_density; end
    if ~isfield(params, 'surface_density'), params.surface_density = default_params.surface_density; end
    if ~isfield(params, 'volume_density'), params.volume_density = default_params.volume_density; end
    if ~isfield(params, 'boundary_density'), params.boundary_density = default_params.boundary_density; end
    if ~isfield(params, 'random_points'), params.random_points = default_params.random_points; end
    if ~isfield(params, 'noise_std'), params.noise_std = default_params.noise_std; end
    if ~isfield(params, 'edge_reduction'), params.edge_reduction = default_params.edge_reduction; end
    if ~isfield(params, 'surface_reduction'), params.surface_reduction = default_params.surface_reduction; end
    if ~isfield(params, 'use_dbscan'), params.use_dbscan = default_params.use_dbscan; end
    if ~isfield(params, 'dbscan_epsilon'), params.dbscan_epsilon = default_params.dbscan_epsilon; end
    if ~isfield(params, 'dbscan_minpts'), params.dbscan_minpts = default_params.dbscan_minpts; end

    if nargin < 5
        visualize = false;
    end

    % edge points generation
    edges = [
             faces(:, [1, 2]);
             faces(:, [2, 3]);
             faces(:, [3, 1])
             ];
    edges = sort(edges, 2);
    edges = unique(edges, 'rows');

    if isempty(gcp('nocreate'))
        parpool('threads');
    end

    edge_indices = randperm(size(edges, 1));
    edge_indices = edge_indices(1:floor(size(edges, 1) / params.edge_reduction));
    edge_points = cell(length(edge_indices), 1);

    parfor idx = 1:length(edge_indices)
        i = edge_indices(idx);
        v1 = vertices(edges(i, 1), :);
        v2 = vertices(edges(i, 2), :);

        edge_length = norm(v2 - v1);
        num_points = max(2, ceil(edge_length * params.edge_density));

        % generate points with slight randomization
        t = linspace(0, 1, num_points)';
        base_points = v1 + t .* (v2 - v1);

        % add small random perturbations
        noise = randn(num_points, 3) * params.noise_std;
        edge_points{idx} = base_points + noise;
    end

    edge_points = cell2mat(edge_points);

    % surface points generation
    face_indices = randperm(size(faces, 1));
    face_indices = face_indices(1:floor(size(faces, 1) / params.surface_reduction));
    surface_points = cell(length(face_indices), 1);

    parfor idx = 1:length(face_indices)
        i = face_indices(idx);
        v1 = vertices(faces(i, 1), :);
        v2 = vertices(faces(i, 2), :);
        v3 = vertices(faces(i, 3), :);

        % calculate triangle area and number of points
        edge1 = v2 - v1;
        edge2 = v3 - v1;
        area = 0.5 * norm(cross(edge1, edge2));
        num_points = max(1, ceil(area * params.surface_density));

        % generate random barycentric coordinates
        r1 = rand(num_points, 1);
        r2 = rand(num_points, 1);

        % convert to barycentric coordinates
        u = 1 - sqrt(r1);
        v = sqrt(r1) .* (1 - r2);
        w = sqrt(r1) .* r2;

        % generate base points
        base_points = u .* v1 + v .* v2 + w .* v3;

        % add small random perturbations
        noise = randn(num_points, 3) * params.noise_std;
        surface_points{idx} = base_points + noise;
    end

    surface_points = cell2mat(surface_points);

    % boundary and volume points generation
    volume_size = diff(env_dims, 1, 2);
    boundary_points = cell(6, 1); % 6 faces of the bounding box

    for i = 1:3

        for j = 1:2
            face_area = prod(volume_size([1:i - 1, i + 1:3]));
            num_points = ceil(face_area * params.boundary_density);

            points = zeros(num_points, 3);
            points(:, i) = env_dims(i, j);

            for k = [1:i - 1, i + 1:3]
                points(:, k) = env_dims(k, 1) + rand(num_points, 1) * volume_size(k);
            end

            % add small inward-facing perturbations
            noise = randn(num_points, 3) * params.noise_std;

            if j == 1
                noise(:, i) = abs(noise(:, i)); % inward for min face
            else
                noise(:, i) = -abs(noise(:, i)); % inward for max face
            end

            idx = (i - 1) * 2 + j;
            boundary_points{idx} = points + noise;
        end

    end

    boundary_points = cell2mat(boundary_points);

    % volume points
    volume = prod(volume_size);
    num_volume_points = ceil(volume * params.volume_density) + params.random_points;

    volume_points = zeros(num_volume_points, 3);

    for i = 1:3
        volume_points(:, i) = env_dims(i, 1) + rand(num_volume_points, 1) * volume_size(i);
    end

    % combine all points
    all_points = [edge_points; surface_points; boundary_points; volume_points];

    if params.use_dbscan

        try
            idx = dbscan(all_points, params.dbscan_epsilon, params.dbscan_minpts);
            valid_points = idx ~= -1;
            all_points = all_points(valid_points, :);
            clusters = unique(idx(idx ~= -1));

            if visualize
                fprintf('DBSCAN Statistics:\n');
                fprintf('Original points: %d\n', size(idx, 1));
                fprintf('Points after clustering: %d\n', sum(valid_points));
                fprintf('Number of clusters: %d\n', length(clusters));
                fprintf('Noise points removed: %d\n', sum(idx == -1));
            end

        catch e
            warning('DBSCAN clustering failed: %s\nProceeding with unclustered points.', e.message);
        end

    end

    if visualize
        figure;

        if params.use_dbscan
            pt_cloud = pointCloud(all_points);
            pcshow(pt_cloud);
            title(sprintf('total points w/ DBSCAN: %d)', size(all_points, 1)));
        else
            pt_cloud = pointCloud(all_points);
            pcshow(pt_cloud);
            title(sprintf('total points %d)', size(all_points, 1)));
        end

        xlabel('x'); ylabel('y'); zlabel('z');
        grid on;
    end

end
```

## File: datasets/get_ray_chan.m
```
function [interaction_points, ray_coeffs] = get_ray_chan(rays, freqs, method)
    % GET_RAY_CHAN Extract channel coefficients from ray tracing results
    %
    % Description:
    %   Computes channel coefficients and interaction points for each ray path
    %   using either Shooting and Bouncing Ray (SBR) method or Free Space
    %   Path Loss (FSPL) calculations.
    %
    % Inputs:
    %   rays   - Array of ray objects from ray tracer
    %   freqs  - Vector of frequencies in Hz for channel computation
    %   method - String specifying computation method: "sbr" or "fspl"
    %
    % Outputs:
    %   interaction_points - [3xN] matrix of final interaction point coordinates for N paths
    %   ray_coeffs         - [FxN] complex matrix of channel coefficients for F frequencies and N paths
    %
    % Example:
    %   rays = rayTrace(tx, rx, environment);
    %   freqs = linspace(2.4e9, 2.5e9, 10);
    %   [interaction_points, ray_coeffs] = get_ray_chan(rays, freqs, "sbr");

    n_paths = length(rays);
    n_freqs = length(freqs);

    interaction_points = zeros(3, n_paths);
    ray_coeffs = zeros(n_freqs, n_paths);

    for i = 1:n_paths

        if rays(i).LineOfSight
            points = [rays(i).TransmitterLocation rays(i).ReceiverLocation];
        else
            points = [rays(i).TransmitterLocation rays(i).Interactions.Location rays(i).ReceiverLocation];
        end

        interaction_points(:, i) = points(:, end - 1);

        for j = 1:n_freqs
            f = freqs(j);

            if strcmp(method, "sbr")
                path_loss = rays(i).PathLoss;
                phase = rays(i).PhaseShift;
            else
                prop_dist = rays(i).PropagationDistance;
                path_loss = fspl(prop_dist, f);
                phase = 2 * pi * f * prop_dist / physconst("lightspeed");
            end

            ray_coeffs(j, i) = 10 ^ (-path_loss / 20) * exp(-1j * phase);
        end
    end
end
```

## File: datasets/ray_marching.m
```
function [step_lengths, ray_points] = ray_marching(rays)
    % RAY_MARCHING Extract ray path information from ray tracing results
    %
    % Description:
    %   Extracts step lengths and interaction points along ray paths from
    %   ray tracing results, handling both line-of-sight and multi-bounce paths.
    %
    % Inputs:
    %   rays - Array of ray objects from ray tracer containing path information
    %
    % Outputs:
    %   step_lengths - Cell array containing lengths of each ray segment
    %   ray_points   - Cell array containing coordinates of ray interaction points
    %
    % Example:
    %   rays = rayTrace(tx, rx, environment);
    %   [step_lengths, ray_points] = ray_marching(rays);

    num_rays = length(rays);
    step_lengths = cell(num_rays, 1);
    ray_points = cell(num_rays, 1);

    for i = 1:num_rays

        if rays(i).LineOfSight
            points = [rays(i).TransmitterLocation rays(i).ReceiverLocation];
        else
            points = [rays(i).TransmitterLocation rays(i).Interactions.Location rays(i).ReceiverLocation];
        end

        ray_points{i} = points;
        point_diffs = diff(points, 1, 2);
        step_lengths{i} = sqrt(sum(point_diffs .^ 2));
    end

end
```

## File: engine/render_magnitude.py
```python
# engine/render_magnitude.py # renamed from render_channel.py

import torch
import torch.nn.functional as F

from models.gaussian_model import GaussianChannelFieldModel


@torch.jit.script  # keep jit for potential performance boost
def compute_spatial_weight(
    d_vec_n: torch.Tensor,  # difference vector (Rx - Gaussian Mean) - shape (N, 3)
    inv_covariance_n: torch.Tensor,  # inverse covariance matrix - shape (N, 3, 3)
    base_activation_n: torch.Tensor,  # base activation (after sigmoid) - shape (N, 1)
    eps: float = 1e-10,
) -> torch.Tensor:
    """Computes the spatial weight alpha_n * GaussianPDF(d_vec_n).

    Args:
        d_vec_n: Difference vectors from Rx pos to Gaussian means (N, 3).
        inv_covariance_n: Inverse covariance matrices for Gaussians (N, 3, 3).
        base_activation_n: Activated base opacities/activations (N, 1).
        eps: Small epsilon for numerical stability (already used in covariance).

    Returns:
        Spatial weights for each Gaussian (N,).
    """
    # add batch dimension for bmm: (N, 1, 3)
    d_vec_n_unsqueezed = d_vec_n.unsqueeze(1)

    # calculate exponent term: -0.5 * (d^T * Sigma^-1 * d)
    # (N, 1, 3) @ (N, 3, 3) -> (N, 1, 3)
    exponent_term = torch.bmm(d_vec_n_unsqueezed, inv_covariance_n)
    # (N, 1, 3) @ (N, 3, 1) -> (N, 1, 1)
    exponent_term = torch.bmm(exponent_term, d_vec_n_unsqueezed.transpose(1, 2))
    # remove extra dims: (N,)
    exponent_term = exponent_term.squeeze()

    # calculate gaussian pdf component (unnormalized - determinant factor ignored as it's constant per gaussian)
    # clamp exponent to avoid exp overflow
    pdf_weight = torch.exp(
        -0.5 * torch.clamp(exponent_term, max=50.0)
    )  # clamp max value

    # multiply by the base activation (opacity)
    # squeeze base_activation_n from (N, 1) to (N,)
    activation_weight = base_activation_n.squeeze(-1)

    # final spatial weight = activation * pdf
    weight = activation_weight * pdf_weight

    # return weight.clamp(min=eps) # clamp minimum weight? maybe not needed if eps handled in cov
    return weight


def render_magnitude(
    rx_positions: torch.Tensor,  # shape (B, 3)
    model: GaussianChannelFieldModel,
    tx_position: torch.Tensor,  # shape (3,) or (1, 3)
    nt: int,
    nr: int,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Renders the normalized channel magnitude matrix H_mag for a batch of receiver positions.

    Args:
        rx_positions: Batch of receiver positions (B, 3).
        model: The GaussianChannelFieldModel instance.
        tx_position: The fixed transmitter position (3,).
        nt: Number of Tx antennas.
        nr: Number of Rx antennas.
        eps: Small value for numerical stability, especially for inverse covariance.

    Returns:
        Batch of predicted normalized channel magnitude matrices (B, Nt, Nr).
    """
    batch_size = rx_positions.shape[0]
    device = rx_positions.device
    tx_position = tx_position.to(device)  # ensure tx pos is on correct device

    # get gaussian parameters
    gauss_means = model.get_xyz  # (N, 3)
    num_gaussians = gauss_means.shape[0]

    # handle case with no gaussians
    if num_gaussians == 0:
        print("Warning: Rendering magnitude with zero Gaussians.")
        return torch.zeros(batch_size, nt, nr, dtype=torch.float32, device=device)

    # get dynamically computed attributes: latent features and base activation *logits*
    gauss_latents, gauss_activation_logits = model.get_attributes(
        tx_position
    )  # (N, latent_dim), (N, 1)
    # activate the base activation logits
    gauss_activations_activated = model.opacity_activation(
        gauss_activation_logits
    )  # (N, 1)

    # get inverse covariance matrices
    _, gauss_inv_covs = model.get_covariance(return_inverse=True, eps=eps)  # (N, 3, 3)

    # --- Prepare for Batch Processing ---
    # expand rx positions and gaussian means for batch calculation
    # rx_pos_expanded: (B, 1, 3)
    rx_pos_expanded = rx_positions.unsqueeze(1)
    # gauss_means_expanded: (1, N, 3)
    gauss_means_expanded = gauss_means.unsqueeze(0)

    # calculate difference vectors d_vec = rx_pos - gauss_mean
    # shape: (B, N, 3)
    d_vec = rx_pos_expanded - gauss_means_expanded

    # flatten d_vec for batch matrix multiplication: (B*N, 3)
    d_vec_flat = d_vec.view(-1, 3)

    # expand inverse covariances and activations for batch processing
    # inv_covs_expanded: (B*N, 3, 3) - repeat each gaussian's cov for every batch item
    inv_covs_expanded = gauss_inv_covs.repeat(batch_size, 1, 1)
    # activations_expanded: (B*N, 1) - repeat each gaussian's activation for every batch item
    activations_expanded = gauss_activations_activated.repeat(batch_size, 1)

    # --- Compute Spatial Weights ---
    # compute weights for all batch-gaussian pairs: (B*N,)
    spatial_weights_flat = compute_spatial_weight(
        d_vec_flat, inv_covs_expanded, activations_expanded, eps
    )
    # reshape back to (B, N)
    spatial_weights = spatial_weights_flat.view(batch_size, num_gaussians)

    # --- Decode Magnitude Contributions ---
    # decode latent features into magnitude contributions (already activated by sigmoid)
    # h_mag_contrib_flat shape: (N, Nt * Nr)
    h_mag_contrib_flat = model.contribution_decoder(gauss_latents)
    # reshape to (N, Nt, Nr)
    h_mag_contrib = h_mag_contrib_flat.view(num_gaussians, nt, nr)

    # --- Combine Contributions ---
    # expand contributions and weights for batch summation
    # h_mag_contrib_expanded: (1, N, Nt, Nr)
    h_mag_contrib_expanded = h_mag_contrib.unsqueeze(0)
    # spatial_weights_expanded: (B, N, 1, 1)
    spatial_weights_expanded = spatial_weights.unsqueeze(-1).unsqueeze(-1)

    # weight the contributions by spatial weights
    # weighted_contributions: (B, N, Nt, Nr)
    weighted_contributions = spatial_weights_expanded * h_mag_contrib_expanded

    # sum contributions over the gaussian dimension (dim=1)
    # H_mag_pred: (B, Nt, Nr)
    H_mag_pred = torch.sum(weighted_contributions, dim=1)

    # ensure output is float32
    return H_mag_pred.float()
```

## File: models/__init__.py
```python
# models/__init__.py

from .gaussian_model import GaussianChannelFieldModel
from .networks import AttributeNetwork, ContributionDecoderNetwork
```

## File: utils/__init__.py
```python
# utils/__init__.py

from .general_utils import *
from .loss import *
from .pos_encoder import *
from .train_utils import *
from .transform_utils import *
```

## File: utils/general_utils.py
```python
# utils/general_utils.py

import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


def set_random_seed(seed: Optional[int] = None) -> None:
    """Set random seeds for reproducibility."""
    if seed is not None:
        print(f"Setting random seed to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
```

## File: utils/pos_encoder.py
```python
# utils/pos_encoder.py

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """Sine-cosine positional encoder for input points."""

    def __init__(
        self,
        input_dims: int,
        num_freqs: int,
        include_input: bool = True,
        log_sampling: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.output_dims = 0
        self.embedding_fns = []
        self._create_embedding_fn()

    def _create_embedding_fn(self):
        """Create the embedding functions."""
        if self.include_input:
            self.embedding_fns.append(lambda x: x)
            self.output_dims += self.input_dims

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(
                0.0, self.num_freqs - 1, steps=self.num_freqs
            )
        else:
            freq_bands = torch.linspace(
                1.0, 2.0 ** (self.num_freqs - 1), steps=self.num_freqs
            )

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                self.embedding_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq)
                )
                self.output_dims += self.input_dims

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        Args:
            inputs: Input tensor (..., input_dims)
        Returns:
            Encoded tensor (..., output_dims)
        """
        encoded = torch.cat([fn(inputs) for fn in self.embedding_fns], dim=-1)
        return encoded
```

## File: utils/train_utils.py
```python
# utils/train_utils.py

import logging
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
) -> Callable[[int], float]:
    """Computes learning rate following exponential decay."""

    def helper(step: int) -> float:
        if step < 0 or max_steps <= 0:
            return 0.0
        if lr_init == 0.0 and lr_final == 0.0:
            return 0.0

        if lr_delay_steps > 0:
            delay_factor = lr_delay_mult + (1.0 - lr_delay_mult) * np.sin(
                0.5 * np.pi * min(step / lr_delay_steps, 1.0)
            )
        else:
            delay_factor = 1.0

        progress = min(step / max_steps, 1.0)

        if lr_init <= 0:
            log_lr_init = -np.inf
        else:
            log_lr_init = np.log(lr_init)

        if lr_final <= 0:
            log_lr_final = -np.inf
        else:
            log_lr_final = np.log(lr_final)

        log_lerped_lr = log_lr_init * (1.0 - progress) + log_lr_final * progress
        lerped_lr = np.exp(log_lerped_lr)

        return delay_factor * lerped_lr

    return helper


def setup_logging(log_dir: Path) -> logging.Logger:
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def compute_grad_stats(
    model: torch.nn.Module, norm_type: float = 2.0
) -> Dict[str, float]:
    """Computes statistics about gradients for model parameters."""
    total_norm = 0.0
    total_abs_sum = 0.0
    total_elements = 0
    min_grad = float("inf")
    max_grad = float("-inf")
    param_count_with_grad = 0

    for param in model.parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(
                    f"Warning: NaN or Inf detected in gradients for parameter. Skipping stats for this param."
                )
                continue

            param_norm = param.grad.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
            total_abs_sum += param.grad.abs().sum().item()
            total_elements += param.numel()
            param_count_with_grad += 1

            current_min = param.grad.min().item()
            current_max = param.grad.max().item()
            min_grad = min(min_grad, current_min)
            max_grad = max(max_grad, current_max)

    total_norm = total_norm ** (1.0 / norm_type) if total_norm > 0 else 0.0
    mean_abs_grad = total_abs_sum / total_elements if total_elements > 0 else 0.0

    if min_grad == float("inf"):
        min_grad = 0.0
    if max_grad == float("-inf"):
        max_grad = 0.0

    return {
        "grad_norm": total_norm,
        "mean_abs_grad": mean_abs_grad,
        "min_grad": min_grad,
        "max_grad": max_grad,
        "param_count_with_grad": param_count_with_grad,
    }
```

## File: utils/transform_utils.py
```python
# utils/transform_utils.py

from typing import Tuple, Union

import torch


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Convert sigmoid output back to logits."""
    eps = torch.finfo(x.dtype).eps
    x = torch.clamp(x, min=eps, max=1.0 - eps)
    return torch.log(x / (1 - x))


@torch.jit.script
def build_rotation(r: torch.Tensor) -> torch.Tensor:
    """Build rotation matrices from quaternions (ensure normalization)."""
    norm = torch.sqrt(torch.sum(r * r, dim=1, keepdim=True)).clamp(min=1e-10)
    q = r / norm

    R = torch.zeros((q.size(0), 3, 3), device=r.device, dtype=r.dtype)

    qw = q[:, 0]
    qx = q[:, 1]
    qy = q[:, 2]
    qz = q[:, 3]

    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz
    qxqy = qx * qy
    qxqz = qx * qz
    qyqz = qy * qz
    qwqx = qw * qx
    qwqy = qw * qy
    qwqz = qw * qz

    R[:, 0, 0] = 1.0 - 2.0 * (qy2 + qz2)
    R[:, 0, 1] = 2.0 * (qxqy - qwqz)
    R[:, 0, 2] = 2.0 * (qxqz + qwqy)
    R[:, 1, 0] = 2.0 * (qxqy + qwqz)
    R[:, 1, 1] = 1.0 - 2.0 * (qx2 + qz2)
    R[:, 1, 2] = 2.0 * (qyqz - qwqx)
    R[:, 2, 0] = 2.0 * (qxqz - qwqy)
    R[:, 2, 1] = 2.0 * (qyqz + qwqx)
    R[:, 2, 2] = 1.0 - 2.0 * (qx2 + qy2)
    return R


@torch.jit.script
def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Build combined scaling and rotation matrix L = R * S."""
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)
    S_diag = torch.diag_embed(s)
    L = R @ S_diag
    return L


@torch.jit.script
def build_covariance_from_scaling_rotation(
    scaling: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Build covariance matrix Σ = R * S^2 * R^T."""
    R = build_rotation(rotation)
    S_sq_diag = torch.diag_embed(scaling * scaling)
    covariance = R @ S_sq_diag @ R.transpose(1, 2)
    return covariance


@torch.jit.script
def build_covariance_inverse(
    R: torch.Tensor, scaling: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Builds the inverse covariance matrix Σ^-1 = R * S^-2 * R^T.
    Handles potential division by zero in scaling.

    Args:
        R: Rotation matrices (N, 3, 3).
        scaling: Activated scaling factors (N, 3).
        eps: Small value to prevent division by zero.

    Returns:
        Inverse covariance matrices (N, 3, 3).
    """
    scaling_clamped = torch.clamp(scaling, min=eps)
    inv_scaling_sq = 1.0 / (scaling_clamped * scaling_clamped)
    S_inv_sq_diag = torch.diag_embed(inv_scaling_sq)

    inv_covariance = R @ S_inv_sq_diag @ R.transpose(1, 2)
    return inv_covariance
```

## File: utils/loss.py
```python
# utils/loss.py

import numpy as np
import torch
import torch.nn as nn

# remove calculate_nmse and NormalizedMSELoss as they are replaced by MSE


def calculate_snr(
    mse_loss: torch.Tensor, target: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """Calculate SNR in dB from MSE loss and target tensor (magnitudes).

    SNR = 10 * log10( Mean(Target^2) / MSE )

    Args:
        mse_loss: The calculated Mean Squared Error loss (scalar tensor).
        target: The ground truth target tensor (e.g., magnitudes).
        eps: Small value for numerical stability, especially for clamping MSE.

    Returns:
        SNR value in dB (scalar tensor). Returns -inf if target power is zero
        or MSE is non-positive after clamping.
    """
    if not isinstance(mse_loss, torch.Tensor) or mse_loss.numel() != 1:
        raise TypeError("mse_loss must be a scalar tensor.")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a tensor.")

    # ensure inputs are on the same device
    target = target.to(device=mse_loss.device)

    # calculate average signal power: Mean(Target^2)
    signal_power = torch.mean(target**2)

    # handle case where signal power is zero or negative (shouldn't happen for magnitude)
    if signal_power <= eps:
        print(
            f"Warning: Target signal power is near zero ({signal_power.item():.2e}). SNR calculation may be unstable or -inf."
        )
        # return -inf or a very small number depending on desired behavior
        return torch.tensor(float("-inf"), device=mse_loss.device)

    # noise power is the MSE loss
    noise_power = mse_loss

    # clamp noise power (MSE) to avoid log10(0) or log10(negative)
    # also handles potential NaN/Inf in mse_loss, though they should ideally be handled earlier
    noise_power_clamped = torch.clamp(noise_power, min=eps)

    # calculate SNR
    snr = 10.0 * torch.log10(signal_power / noise_power_clamped)

    # check for final NaN/Inf (e.g., if signal_power / noise_power_clamped is negative or zero)
    if torch.isnan(snr) or torch.isinf(snr):
        print(
            f"Warning: Final SNR is NaN or Inf (Signal Power: {signal_power.item():.2e}, Clamped MSE: {noise_power_clamped.item():.2e}). Returning -Inf."
        )
        return torch.tensor(float("-inf"), device=mse_loss.device)

    return snr


# Note: Consider adding a standard MSELoss class wrapper if needed elsewhere,
# but usually `torch.nn.MSELoss()` is used directly.
# class MSELoss(nn.Module):
#     """Computes the Mean Squared Error (MSE) loss."""
#     def __init__(self, reduction: str = 'mean'):
#         super().__init__()
#         self.mse = nn.MSELoss(reduction=reduction)

#     def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             pred: Predicted tensor.
#             target: Ground truth tensor.
#         Returns:
#             Scalar MSE loss.
#         """
#         # ensure target is on the same device and type
#         target = target.to(device=pred.device, dtype=pred.dtype)
#         return self.mse(pred, target)
```

## File: datasets/generate_csi.m
```
function [H, AoD, AoA] = generate_csi(rays, fc, cfg, txArray, rxArray, method, scenario, use_single_sc, sc_idx)
    % GENERATE_CSI Generate a spatially-consistent, frequency-selective MIMO channel
    %
    % Description:
    %   Generates a MIMO channel tensor H from ray tracing data.
    %   Computes the channel response for each subcarrier in the OFDM/NR signal.
    %   The channel tensor is computed using the steering vectors of the transmit
    %   and receive arrays, and the path loss and phase shift of each ray.
    %
    % Inputs:
    %   rays          - Ray objects array from ray tracing
    %   fc            - Center frequency (Hz)
    %   cfg           - OFDM/NR carrier configuration struct
    %   txArray       - Transmit array (phased.URA or phased.IsotropicAntennaElement)
    %   rxArray       - Receive array (phased.ULA or phased.IsotropicAntennaElement)
    %   method        - 'sbr' to use ray.PathLoss/PhaseShift, 'fspl' for pure FSPL
    %   scenario      - "indoor" or "outdoor"
    %   use_single_sc - true to pick one subcarrier via sc_idx, false for all
    %   sc_idx        - Subcarrier index if use_single_sc==true
    %
    % Outputs:
    %   H   - Nt x Nr x Nsc channel tensor (complex)
    %   AoD - 2 x Nrays matrix of departure [az;el] in degrees
    %   AoA - 2 x Nrays matrix of arrival   [az;el] in degrees
    %
    % Example:
    %   [H, AoD, AoA] = generate_csi(rays, fc, cfg, txArray, rxArray, 'sbr', "outdoor", true, 1);

    if nargin < 8, use_single_sc = false; end
    if nargin < 9, sc_idx = []; end

    is_siso = isa(txArray, 'phased.IsotropicAntennaElement') && isa(rxArray, 'phased.IsotropicAntennaElement');

    if scenario == "indoor"
        ofdmInfo   = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
        actIdx     = ofdmInfo.ActiveFrequencyIndices;
        sc_sp      = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;
        if use_single_sc
            if isempty(sc_idx), sc_idx = ceil(numel(actIdx)/2); end
            freqs = fc + actIdx(sc_idx)*sc_sp;
        else
            freqs = fc + actIdx*sc_sp;
        end
    else
        sc_sp      = cfg.SubcarrierSpacing * 1e3;
        totalSC    = cfg.NSizeGrid * 12;
        actIdx     = -totalSC/2 : totalSC/2-1;
        if use_single_sc
            if isempty(sc_idx), sc_idx = ceil(numel(actIdx)/2); end
            freqs = fc + actIdx(sc_idx)*sc_sp;
        else
            freqs = fc + actIdx*sc_sp;
        end
    end
    Nsc = numel(freqs);

    if is_siso
        Nt = 1;
        Nr = 1;
    else
        Nt = prod(txArray.Size);
        Nr = rxArray.NumElements;
    end
    H  = zeros(Nt, Nr, Nsc);

    if ~is_siso
        svTx = phased.SteeringVector( ...
            'SensorArray',            txArray, ...
            'PropagationSpeed',       physconst('LightSpeed'), ...
            'IncludeElementResponse', false );
        svRx = phased.SteeringVector( ...
            'SensorArray',            rxArray, ...
            'PropagationSpeed',       physconst('LightSpeed'), ...
            'IncludeElementResponse', false );
    end

    numRays = numel(rays);
    AoD     = zeros(2, numRays);
    AoA     = zeros(2, numRays);

    for r = 1:numRays
        ray  = rays(r);
        dist = ray.PropagationDistance;
        tau  = dist / physconst('LightSpeed');

        if ray.LineOfSight
            pts = [ray.TransmitterLocation, ray.ReceiverLocation];
        else
            pts = [ray.TransmitterLocation, ray.Interactions.Location, ray.ReceiverLocation];
        end

        % departure
        vecTx      = pts(:,2) - pts(:,1);
        [azT, elT] = cart2sph(vecTx(1), vecTx(2), vecTx(3));
        azT = rad2deg(azT);  elT = rad2deg(elT);
        AoD(:,r) = [azT; elT];

        % arrival
        vecRx      = pts(:,end) - pts(:,end-1);
        [azR, elR] = cart2sph(vecRx(1), vecRx(2), vecRx(3));
        azR = rad2deg(azR);  elR = rad2deg(elR);
        AoA(:,r) = [azR; elR];

        if ~is_siso
            aTx = svTx(fc, [azT; elT]);
            aRx = svRx(fc, [azR; elR]);
            array_gain = aTx * aRx.';
        else
            array_gain = 1; % isotropic antennas have gain of 1 (0 dBi)
        end

        for k = 1:Nsc
            f = freqs(k);
            if strcmp(method, 'sbr')
                pl0   = ray.PathLoss;
                phi0  = ray.PhaseShift;
                pl    = pl0;
                phase = phi0 + 2*pi*(f - fc)*tau;
            else
                pl    = fspl(dist, f);
                phase = 2*pi*f*tau;
            end
            h = 10^(-pl/20) * exp(-1j*phase);
            H(:,:,k) = H(:,:,k) + array_gain * h;
        end
    end
end
```

## File: datasets/outdoor.m
```
%% environment setup
close all force; clear; clc;
plot_rays = false;
visualize = false;

% load both file formats
stl_file = "models/intersection_and_buildings.stl";
mapFileName = "models/intersection_and_buildings/IntersectionAndBuildings.glb";
[stl_data, ~] = stlread(stl_file);

if visualize
    viewer = siteviewer("SceneModel", mapFileName, "ShowEdges", false, "ShowOrigin", false);
end

% environment dimensions setup from stl
vertices = stl_data.Points;
faces = stl_data.ConnectivityList;

xy_offset = 0.1;
z_offset = 0.1;
min_z = max(0, min(vertices(:, 3)));
env_dims = [
            [min(vertices(:, 1)) + xy_offset, max(vertices(:, 1)) - xy_offset];
            [min(vertices(:, 2)) + xy_offset, max(vertices(:, 2)) - xy_offset];
            [min_z, max(vertices(:, 3)) - z_offset]
            ];

% point cloud generation params
pc_params = struct();
pc_params.edge_density = 0.43;
pc_params.surface_density = 0.06;
pc_params.volume_density = 0;
pc_params.boundary_density = 0;
pc_params.random_points = 0;
pc_params.noise_std = 0;
pc_params.edge_reduction = 1;
pc_params.surface_reduction = 1;

%% generate point cloud
point_cloud = generate_pc(vertices, faces, pc_params, env_dims, visualize);

%% System config
fc = 6e9;
lambda = physconst("lightspeed") / fc;

% OFDM parameters
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 15;
carrier.NSizeGrid = 52;
cfg = carrier;

% extra config
use_single_sc = true;
sc_idx = [];
use_siso = false;

if use_siso
    % single-input single-output
    txArray = phased.IsotropicAntennaElement();
    rxArray = phased.IsotropicAntennaElement();

    num_tx_ant = 1;
    num_rx_ant = 1;
else
    % multiple-input multiple-output
    txArray = phased.URA("Size", [8 8], "ElementSpacing", lambda / 2);
    rxArray = phased.ULA("NumElements", 2, "ElementSpacing", lambda / 2);

    num_tx_ant = prod(txArray.Size);
    num_rx_ant = rxArray.NumElements;
end

%% AP setup
AP = txsite("cartesian", ...
    "Antenna", txArray, ...
    "AntennaPosition", [20; 38; 20], ...
    "TransmitterFrequency", fc, ...
    "TransmitterPower", 10);

%% user setup
approx_target_users = 2316;

% seed
S = RandStream("mt19937ar", "Seed", 17);
RandStream.setGlobalStream(S);

user_params.check_building_collision = true;
user_params.check_user_collision = true;
user_params.separation_distance = 1;
[Users, actual_users] = create_users(env_dims, approx_target_users, rxArray, user_params, []);

if actual_users < approx_target_users
    error('Failed to create all requested users. Only created %d out of %d users.', ...
        actual_users, approx_target_users);
end

%% RT simulation
method = "sbr"; % "image" | "sbr"
max_refs = 2;

pm = propagationModel("raytracing", ...
    "Method", method, ...
    "CoordinateSystem", "cartesian", ...
    "MaxNumDiffractions", 1, ...
    "MaxNumReflections", max_refs, ...
    "UseGPU", "on");

rays = raytrace(AP, Users, pm, "Map", mapFileName, "Type", "pathloss");

% filter users to keep only those with valid rays
valid_user_mask = ~cellfun(@isempty, rays);
Users = Users(valid_user_mask);
rays = rays(valid_user_mask);
num_users = sum(valid_user_mask);

if num_users < approx_target_users
    warning('Only %d out of %d users had valid rays, discarding the rest.', ...
        num_users, approx_target_users);
end

%% visualize
if visualize
    show(AP, "ShowAntennaHeight", false)
    show(Users, "ShowAntennaHeight", false)

    if plot_rays
        for userIdx = 1:(num_users / 20) % ~20 % rays
            plot(rays{userIdx}, "Colormap", jet)
            pause(0.05)
        end
    end
end

%% extract positions
rx_positions = zeros(3, num_users);

for userIdx = 1:num_users
    rx_positions(:, userIdx) = Users(userIdx).AntennaPosition;
end

%% CSI collection
if use_single_sc
    numSubcarriers = 1;
    sc_spacing = cfg.SubcarrierSpacing * 1e3;
    total_scs = cfg.NSizeGrid * 12;
    activeFreqIndices = (-total_scs / 2:total_scs / 2 - 1);
    
    if isempty(sc_idx)
        sc_idx = ceil(length(activeFreqIndices) / 2);
    end
    freqs = fc + activeFreqIndices(sc_idx) * sc_spacing;
else
    numSubcarriers = cfg.NSizeGrid * 12;
    sc_spacing = cfg.SubcarrierSpacing * 1e3;
    activeFreqIndices = (-numSubcarriers / 2:numSubcarriers / 2 - 1);
    freqs = fc + activeFreqIndices * sc_spacing;
end

H = zeros(num_users, num_tx_ant, num_rx_ant, numSubcarriers);
AoD_all = cell(num_users, 1);
AoA_all = cell(num_users, 1);
path_loss = zeros(num_users, 1);
path_loss_per_ray = cell(num_users, 1);

for userIdx = 1:num_users
    [H(userIdx, :, :, :), AoD_all{userIdx}, AoA_all{userIdx}] = ...
        generate_csi(rays{userIdx}, fc, cfg, txArray, rxArray, method, 'outdoor', use_single_sc, sc_idx);
    path_loss(userIdx) = mean([rays{userIdx}.PathLoss]);
    path_loss_per_ray{userIdx} = [rays{userIdx}.PathLoss];
end

% check for null values in channel matrix
if any(isnan(H(:)))
    warning('Channel matrix contains NaN values!');
end

%% ray marching and per-ray information (for NeWRF comparison)
ray_steps = cell(num_users, 1);
ray_points = cell(num_users, 1);
ray_interactions = cell(num_users, 1);
ray_coefficients = cell(num_users, 1);

for userIdx = 1:num_users
    [ray_steps{userIdx}, ray_points{userIdx}] = ray_marching(rays{userIdx});
    [ray_interactions{userIdx}, ray_coefficients{userIdx}] = get_ray_chan(rays{userIdx}, freqs, method);
end

% check for null values in ray marching results
if any(cellfun(@(x) any(cellfun(@(y) any(isnan(y(:))), x)), ray_steps)) || ...
        any(cellfun(@(x) any(cellfun(@(y) any(isnan(y(:))), x)), ray_points)) || ...
        any(cellfun(@(x) any(isnan(x(:))), ray_coefficients))
    warning('Ray marching results contain NaN values!');
end

%% save
output_dir = "outputs";
mkdir(output_dir);

[~, mapname] = fileparts(mapFileName);

% create dataset
dataset = struct();

dataset.config.tx_antennas = num_tx_ant;
dataset.config.rx_antennas = num_rx_ant;
dataset.config.frequency = fc;
dataset.config.wavelength = lambda;
dataset.config.num_users = num_users;
dataset.config.use_siso = use_siso;

dataset.environment.dimensions = env_dims;
dataset.environment.point_cloud = point_cloud;
dataset.environment.pc_params = pc_params;

dataset.nodes.ap_position = AP.AntennaPosition';
dataset.nodes.users_positions = rx_positions;

dataset.channel.H = H;
% NOTE: AoD is unneeded as it is primarily related to the txsite
% dataset.channel.AoD = AoD_all;
dataset.channel.path_loss = path_loss;

% truncate data (max 10 paths)
[AoA_trunc, AoD_trunc, path_loss_per_ray_trunc] = truncate_data(AoA_all, AoD_all, path_loss_per_ray, 10);
dataset.channel.AoA = AoA_trunc;
dataset.channel.AoD = AoD_trunc;
dataset.channel.path_loss_per_ray = path_loss_per_ray_trunc;

dataset.channel.ray_steps = ray_steps;
dataset.channel.ray_points = ray_points;
dataset.channel.ray_interactions = ray_interactions;
dataset.channel.ray_coefficients = ray_coefficients;
dataset.channel.frequencies = freqs;

sc_str = '';

if use_single_sc
    sc_str = sprintf('_sc%d', sc_idx);
end

filename = sprintf('%s/iab_%dx%d_%du_%.1fghz_%sRT%s.mat', ...
    output_dir, ...
    num_tx_ant, ...
    num_rx_ant, ...
    num_users, ...
    fc / 1e9, ...
    method, ...
    sc_str);

save(filename, 'dataset', '-v7.3');

function [AoA_trunc, AoD_trunc, path_loss_trunc] = truncate_data(AoA, AoD, path_loss, max_paths)
    num_users = length(AoA);
    AoA_trunc = cell(num_users, 1);
    AoD_trunc = cell(num_users, 1);
    path_loss_trunc = cell(num_users, 1);
    
    for i = 1:num_users
        if ~isempty(AoA{i})
            % sort paths by path loss
            [sorted_pl, sort_idx] = sort(path_loss{i});
            sorted_AoA = AoA{i}(:, sort_idx);
            sorted_AoD = AoD{i}(:, sort_idx);
            
            % take top max_paths with lowest path loss
            num_paths = min(length(sorted_pl), max_paths);
            AoA_trunc{i} = sorted_AoA(:, 1:num_paths);
            AoD_trunc{i} = sorted_AoD(:, 1:num_paths);
            path_loss_trunc{i} = sorted_pl(1:num_paths);
        else
            AoA_trunc{i} = [];
            AoD_trunc{i} = [];
            path_loss_trunc{i} = [];
        end
    end
end
```

## File: engine/__init__.py
```python
# engine/__init__.py

# update the import to reflect the new module name and function
from .render_magnitude import render_magnitude
```

## File: models/networks.py
```python
# models/networks.py

from typing import Optional, Tuple

import torch
import torch.nn as nn

from utils.pos_encoder import PositionalEncoder


class SimpleMLP(nn.Module):
    """A simple Multi-Layer Perceptron"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        use_leaky_relu: bool = True,  # leaky relu often works well
        dropout_p: float = 0.1,  # moderate dropout
        final_activation: Optional[nn.Module] = None,  # allow final activation
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  # total layers including output

        if num_layers < 2:
            raise ValueError("MLP must have at least 2 layers (input -> output)")

        layers = []
        current_dim = input_dim

        # hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            # use leaky relu or gelu for hidden layers
            layers.append(nn.LeakyReLU(0.1) if use_leaky_relu else nn.GELU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            current_dim = hidden_dim

        # output layer
        layers.append(nn.Linear(current_dim, output_dim))

        # add final activation if specified
        if final_activation is not None:
            layers.append(final_activation)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ContributionDecoderNetwork(SimpleMLP):
    """
    Decodes latent features into channel magnitude contributions (normalized).
    Output dimension is Nt * Nr. Includes a final Sigmoid activation.
    """

    def __init__(
        self, latent_dim: int, output_dim: int, hidden_dim: int, num_layers: int
    ):
        # output_dim should be Nt * Nr
        super().__init__(
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_leaky_relu=True,  # leaky relu for hidden layers
            dropout_p=0.1,  # moderate dropout
            final_activation=nn.Sigmoid(),  # sigmoid to output [0, 1] normalized magnitude
        )


class AttributeNetwork(nn.Module):
    """
    Predicts latent features and base activations from
    Gaussian position and fixed Tx position using positional encoding.
    """

    def __init__(
        self,
        latent_dim: int,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        pos_encoding_freqs: int = 10,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # positional encoders for gaussian mean and tx position
        self.pos_encoder_mean = PositionalEncoder(
            input_dims=3, num_freqs=pos_encoding_freqs, include_input=True
        )
        self.pos_encoder_tx = PositionalEncoder(
            input_dims=3, num_freqs=pos_encoding_freqs, include_input=True
        )

        encoded_dim_mean = self.pos_encoder_mean.output_dims
        encoded_dim_tx = self.pos_encoder_tx.output_dims
        # concatenated input dimension for the MLP
        input_dim = encoded_dim_mean + encoded_dim_tx

        # output dimension is latent_dim + 1 (for base activation logit)
        output_dim = latent_dim + 1

        # main MLP network
        self.network = SimpleMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            use_leaky_relu=True,  # use leaky relu in attribute net as well
            dropout_p=0.0,  # typically no dropout here, but could be added
            final_activation=None,  # no final activation needed here
        )

    def forward(
        self, mu_n: torch.Tensor, p_tx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts latent features and base activation logits.

        Args:
            mu_n: Gaussian means (N, 3)
            p_tx: Transmitter position (1, 3) or (3,) - will be expanded

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Latent features (N, latent_dim)
                - Base activation logits (N, 1)
        """
        num_gaussians = mu_n.shape[0]
        if num_gaussians == 0:
            return torch.empty(0, self.latent_dim, device=mu_n.device), torch.empty(
                0, 1, device=mu_n.device
            )

        # ensure p_tx is (N, 3)
        if p_tx.dim() == 1:
            p_tx_expanded = p_tx.unsqueeze(0).expand(num_gaussians, -1)
        elif p_tx.shape[0] == 1:
            p_tx_expanded = p_tx.expand(num_gaussians, -1)
        elif p_tx.shape[0] == num_gaussians:
            p_tx_expanded = p_tx  # already expanded
        else:
            raise ValueError(
                f"p_tx shape {p_tx.shape} incompatible with mu_n shape {mu_n.shape}"
            )

        # encode positions
        encoded_mu = self.pos_encoder_mean(mu_n)
        encoded_ptx = self.pos_encoder_tx(p_tx_expanded)

        # concatenate and pass through MLP
        mlp_input = torch.cat([encoded_mu, encoded_ptx], dim=-1)
        output = self.network(mlp_input)

        # split output into latent features and base activation logits
        latent_features = output[:, : self.latent_dim]
        base_activations_logits = output[:, self.latent_dim :]  # shape (N, 1)

        return latent_features, base_activations_logits
```

## File: datasets/wireless_dataset.py
```python
# datasets/wireless_dataset.py
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from pymatreader import read_mat
from torch.utils.data import Dataset, random_split


class WirelessDataset(Dataset):
    """A dataset class for MIMO/SISO channel magnitude data.

    Handles loading and processing of wireless channel data from .mat files,
    extracting channel magnitude, and applying min-max normalization across
    the entire dataset. Point cloud and env_dims are loaded if present.

    Args:
        data_path (str): Path to the .mat dataset file
        train (bool, optional): If True, returns training set, else test set
        train_ratio (float, optional): Ratio of data for training (default: 0.8)
        seed (int, optional): Random seed for train/test split
        norm_eps (float, optional): Epsilon for normalization denominator stability
    """

    def __init__(
        self,
        data_path: str,
        train: bool = True,
        train_ratio: float = 0.8,
        seed: Optional[int] = None,
        norm_eps: float = 1e-8,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.seed = seed if seed is not None else 42
        self.norm_eps = norm_eps
        generator = torch.Generator().manual_seed(self.seed)
        np.random.seed(self.seed)  # ensure numpy uses seed too if needed elsewhere

        print(f"Loading dataset from: {self.data_path}")
        try:
            mat_data = read_mat(str(self.data_path))
            if "dataset" not in mat_data:
                raise KeyError("Loaded .mat file does not contain 'dataset' key.")
            data = mat_data["dataset"]
        except Exception as e:
            print(f"Error loading MAT file: {e}")
            raise

        print("Processing dataset...")
        self._process_data(data)
        print("Dataset processing complete.")

        total_size = self.num_users
        if total_size == 0:
            raise ValueError("Dataset contains no users/samples.")

        train_size = int(total_size * train_ratio)
        test_size = total_size - train_size

        if train_size == 0 or test_size == 0:
            print(
                f"Warning: train_ratio {train_ratio} resulted in zero samples for train/test split. Adjusting."
            )
            if total_size >= 2:
                train_size = max(1, train_size)
                test_size = total_size - train_size
            else:  # total_size == 1
                train_size = 1 if train else 0
                test_size = 1 - train_size
            print(f"Adjusted split: Train={train_size}, Test={test_size}")

        self.indices = list(range(total_size))
        # use torch.randperm for splitting if generator is needed consistently
        # indices_perm = torch.randperm(total_size, generator=generator).tolist()
        # train_indices = indices_perm[:train_size]
        # test_indices = indices_perm[train_size:]
        # using random_split is simpler if exact indices aren't needed elsewhere
        train_indices_dataset, test_indices_dataset = random_split(
            range(total_size), [train_size, test_size], generator=generator
        )
        self.train_indices = train_indices_dataset.indices
        self.test_indices = test_indices_dataset.indices

        self.active_indices = self.train_indices if train else self.test_indices
        print(f"{'Training' if train else 'Test'} set size: {len(self.active_indices)}")
        if not self.active_indices:
            print(f"Warning: {'Training' if train else 'Test'} set is empty!")

    def _process_data(self, data):
        """Extracts, processes, and normalizes data from the loaded dictionary."""
        self._store_config(data["config"])

        # load environment data if available
        self.point_cloud_data = None
        self.env_dims = None
        if "environment" in data:
            if "point_cloud" in data["environment"]:
                self.point_cloud_data = torch.from_numpy(
                    data["environment"]["point_cloud"]
                ).float()
            if "dimensions" in data["environment"]:
                self.env_dims = torch.from_numpy(
                    data["environment"]["dimensions"]
                ).float()

        # load node positions
        if "nodes" in data and "ap_position" in data["nodes"]:
            tx_pos_raw = data["nodes"]["ap_position"]
            self.tx_position = torch.from_numpy(np.array(tx_pos_raw)).float().squeeze()
            if self.tx_position.shape != (3,):
                raise ValueError(
                    f"Unexpected transmitter position shape: {self.tx_position.shape}, expected (3,)"
                )
        else:
            raise ValueError(
                "Transmitter position ('ap_position') not found in dataset."
            )

        if "nodes" in data and "users_positions" in data["nodes"]:
            rx_pos_raw = data["nodes"]["users_positions"]
            # expect shape (3, K), transpose to (K, 3)
            if rx_pos_raw.ndim == 2 and rx_pos_raw.shape[0] == 3:
                self.rx_positions = torch.from_numpy(rx_pos_raw.T).float()
            else:
                raise ValueError(
                    f"Expected receiver positions shape (3, K), got {rx_pos_raw.shape}"
                )

            num_users_from_pos = self.rx_positions.shape[0]
            if num_users_from_pos != self.num_users:
                print(
                    f"Warning: num_users mismatch. Config: {self.num_users}, Rx Positions: {num_users_from_pos}. Using {num_users_from_pos}."
                )
                self.num_users = num_users_from_pos
        else:
            raise ValueError(
                "Receiver positions ('users_positions') not found in dataset."
            )

        # load and process channel matrix H
        if "channel" in data and "H" in data["channel"]:
            H_raw = data["channel"]["H"]
            # handle complex struct format from matlab
            if isinstance(H_raw, dict) and "real" in H_raw and "imag" in H_raw:
                H_real = np.array(H_raw["real"])
                H_imag = np.array(H_raw["imag"])
                if H_real.dtype.kind not in "iufc" or H_imag.dtype.kind not in "iufc":
                    raise TypeError("Real/Imag parts of H are not numeric.")
                # ensure correct type casting before complex creation
                H_complex = H_real.astype(np.float32) + 1j * H_imag.astype(np.float32)
                H_tensor = torch.from_numpy(H_complex).to(torch.complex64)
            elif isinstance(H_raw, np.ndarray) and np.iscomplexobj(H_raw):
                H_tensor = torch.from_numpy(H_raw).to(torch.complex64)
            else:
                raise TypeError(
                    f"Unsupported format for H: {type(H_raw)}. Expected complex numpy array or dict with 'real'/'imag'."
                )

            print(f"Raw H tensor shape from MAT: {H_tensor.shape}")

            # --- Reshape H tensor ---
            # target shape (num_users, Nt, Nr)
            expected_leading_dim = self.num_users
            target_shape = (expected_leading_dim, self.num_tx_ant, self.num_rx_ant)

            if H_tensor.dim() == 4:  # (N_user, Nt, Nr, N_sc)
                num_sc = H_tensor.shape[-1]
                # select middle subcarrier if multiple exist
                sc_idx = num_sc // 2
                print(
                    f"Multiple subcarriers ({num_sc}) detected. Selecting middle subcarrier index {sc_idx}."
                )
                # slice first N_user samples correctly
                H_selected = H_tensor[:expected_leading_dim, :, :, sc_idx]

            elif H_tensor.dim() == 3:  # (N_user, Nt, Nr)
                H_selected = H_tensor[:expected_leading_dim, :, :]

            elif H_tensor.dim() == 1:  # (N_user,) - possible SISO case
                if self.is_siso:
                    H_selected = H_tensor[:expected_leading_dim]
                else:
                    raise ValueError(
                        f"H tensor has dim 1, but config is MIMO (Nt={self.num_tx_ant}, Nr={self.num_rx_ant})."
                    )

            elif (
                H_tensor.dim() == 2
            ):  # (N_user, Nr) or (N_user, Nt) or maybe (N_user, N_sc)?
                # handle SISO case (N_user, 1)
                if self.is_siso and H_tensor.shape[1] == 1:
                    H_selected = H_tensor[:expected_leading_dim, :]
                # handle potential flattened MIMO (N_user, Nt*Nr) - less likely from generation script
                elif (
                    H_tensor.shape[0] == expected_leading_dim
                    and H_tensor.shape[1] == self.num_tx_ant * self.num_rx_ant
                ):
                    print(
                        f"Warning: H tensor has shape {H_tensor.shape}. Assuming flattened MIMO and reshaping."
                    )
                    H_selected = H_tensor[:expected_leading_dim, :]
                else:
                    raise ValueError(
                        f"Ambiguous H tensor shape {H_tensor.shape} for MIMO/SISO config."
                    )
            else:
                raise ValueError(
                    f"Unexpected H tensor dimensions: {H_tensor.dim()}. Expected 1, 2, 3, or 4."
                )

            # ensure H_selected has the target shape
            try:
                H_reshaped = H_selected.reshape(target_shape)
            except RuntimeError as e:
                print(
                    f"Error reshaping H from {H_selected.shape} to {target_shape}: {e}"
                )
                raise ValueError(
                    f"Could not reshape H tensor to target shape {target_shape}."
                )

            print(f"Reshaped H tensor shape: {H_reshaped.shape}")

            # --- Calculate Magnitude ---
            self.channel_magnitude_raw = torch.abs(H_reshaped)
            print(f"Raw magnitude tensor shape: {self.channel_magnitude_raw.shape}")

            # --- Calculate Normalization Parameters (Min/Max) ---
            # compute over the *entire dataset* before splitting
            self.min_magnitude = torch.min(self.channel_magnitude_raw)
            self.max_magnitude = torch.max(self.channel_magnitude_raw)
            print(
                f"Magnitude range (min/max): {self.min_magnitude:.4e} / {self.max_magnitude:.4e}"
            )

            # --- Apply Normalization ---
            if (self.max_magnitude - self.min_magnitude) < self.norm_eps:
                print(
                    f"Warning: Magnitude range is very small ({self.max_magnitude - self.min_magnitude:.2e}). Setting normalized magnitude to 0.5."
                )
                self.channel_magnitude_normalized = torch.full_like(
                    self.channel_magnitude_raw, 0.5
                )
            else:
                self.channel_magnitude_normalized = (
                    self.channel_magnitude_raw - self.min_magnitude
                ) / (self.max_magnitude - self.min_magnitude + self.norm_eps)
                # clamp to [0, 1] just in case eps causes slight overshoot
                self.channel_magnitude_normalized = torch.clamp(
                    self.channel_magnitude_normalized, 0.0, 1.0
                )

            print(
                f"Normalized magnitude tensor shape: {self.channel_magnitude_normalized.shape}"
            )
            print(
                f"Normalized magnitude range (min/max): {torch.min(self.channel_magnitude_normalized):.4f} / {torch.max(self.channel_magnitude_normalized):.4f}"
            )

        else:
            raise ValueError("Channel matrix ('H') not found in dataset.")

    def _store_config(self, config):
        """Stores configuration values from the dataset and infers SISO."""
        try:
            self.num_tx_ant = int(config["tx_antennas"])
            self.num_rx_ant = int(config["rx_antennas"])
            self.frequency = float(config["frequency"])
            self.wavelength = float(config["wavelength"])
            self.num_users = int(config["num_users"])
            # handle missing 'use_siso' key gracefully
            self.config_use_siso = config.get("use_siso", False)

            # determine if SISO based on antenna counts OR the config flag
            self.is_siso = (
                self.num_tx_ant == 1 and self.num_rx_ant == 1
            ) or self.config_use_siso
            if self.is_siso:
                # enforce SISO antenna counts if flag is true or counts imply it
                self.num_tx_ant = 1
                self.num_rx_ant = 1

            print(
                f"Dataset Config: Nt={self.num_tx_ant}, Nr={self.num_rx_ant}, Freq={self.frequency/1e9:.2f}GHz, Lambda={self.wavelength:.4f}m, NumUsers={self.num_users}, IsSISO={self.is_siso}"
            )

        except KeyError as e:
            print(f"Error: Missing key in dataset config: {e}")
            raise
        except (ValueError, TypeError) as e:
            print(f"Error: Invalid value or type in dataset config: {e}")
            raise

    def get_point_cloud(self) -> Optional[torch.Tensor]:
        """Get the loaded point cloud data if available."""
        return self.point_cloud_data

    def get_env_dims(self) -> Optional[torch.Tensor]:
        """Get environment dimensions if available."""
        return self.env_dims

    def get_tx_position(self) -> torch.Tensor:
        """Get the transmitter position."""
        return self.tx_position

    def get_metadata(self) -> dict:
        """Returns essential metadata including normalization parameters."""
        return {
            "num_tx_ant": self.num_tx_ant,
            "num_rx_ant": self.num_rx_ant,
            "frequency": self.frequency,
            "wavelength": self.wavelength,
            "is_siso": self.is_siso,
            "tx_position": self.tx_position,
            "env_dims": self.env_dims,
            "point_cloud": self.point_cloud_data,
            "min_magnitude": self.min_magnitude,
            "max_magnitude": self.max_magnitude,
            "norm_eps": self.norm_eps,
        }

    def __len__(self):
        return len(self.active_indices)

    def __getitem__(self, idx):
        """Retrieves a single sample (normalized magnitude) for the active split."""
        # map the index relative to the active split to the original index
        original_idx = self.active_indices[idx]
        return {
            "rx_position": self.rx_positions[original_idx],
            "channel_magnitude": self.channel_magnitude_normalized[original_idx],
            "index": original_idx,  # return original index for reference
        }
```

## File: eval.py
```python
# eval.py

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn  # import nn for MSELoss
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from datasets.dataloader import get_wireless_dataloader

# import updated rendering engine
from engine.render_magnitude import render_magnitude
from models.gaussian_model import GaussianChannelFieldModel
from utils.general_utils import set_random_seed

# import updated loss and snr calculation
from utils.loss import calculate_snr  # MSELoss is used directly


def setup_eval_logging(
    log_dir: Path, checkpoint_name: str
) -> Tuple[logging.Logger, Console]:
    """Setup logging configuration for evaluation."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eval_{checkpoint_name}_{timestamp}.log"

    # create logger instance
    logger = logging.getLogger("EvaluationLogger")
    logger.setLevel(logging.INFO)  # set logging level

    # prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # file handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # console handler (rich)
    console = Console(log_path=False)  # prevent rich from writing its own log file
    console_handler = RichHandler(
        console=console, rich_tracebacks=True, markup=True, show_path=False
    )
    console_handler.setFormatter(
        logging.Formatter("%(message)s")
    )  # simpler format for console
    logger.addHandler(console_handler)

    logger.info(f"Evaluation logging initialized. Log file: {log_file}")
    return logger, console


def parse_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Gaussian Channel Field Model for Magnitude Prediction"
    )

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/eval_magnitude",
        help="Directory to save evaluation logs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers (default: 0)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num_samples_to_log",
        type=int,
        default=5,
        help="Number of sample predictions to log in detail",
    )
    # parser.add_argument("--loss_eps", type=float, default=1e-10, help="Epsilon for SNR calculation stability") # Renamed
    parser.add_argument(
        "--snr_calc_eps",
        type=float,
        default=1e-10,
        help="Epsilon for SNR calculation stability",
    )
    parser.add_argument(
        "--norm_eps",
        type=float,
        default=1e-8,
        help="Epsilon used for dataset normalization (must match training)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train ratio used during training (to split test set correctly)",
    )
    parser.add_argument(
        "--unnormalize_samples",
        action="store_true",
        help="Un-normalize sample predictions/targets before logging",
    )

    args = parser.parse_args()
    return args


def format_magnitude_tensor(tensor: torch.Tensor) -> str:
    """Formats a magnitude tensor (real numbers) for logging."""
    if torch.is_complex(tensor):
        # This shouldn't happen if evaluation is correct, but handle defensively
        print("Warning: format_magnitude_tensor received a complex tensor.")
        tensor = torch.abs(tensor)

    # format numpy array with fixed precision
    formatted = np.array2string(
        tensor.cpu().numpy(),
        formatter={"float_kind": lambda x: f"{x:.6f}"},  # format floats
        separator=", ",
    )
    return formatted


def display_stats_table(console: Console, stats: Dict) -> None:
    """Display statistics in a rich table."""
    table = Table(title="Evaluation Statistics")

    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right", style="green")
    table.add_column("Std Dev", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Count", justify="right")

    # add MSE loss row
    table.add_row(
        "MSE Loss (Normalized)",
        f"{stats['loss']['mean']:.6e}",
        f"{stats['loss']['std']:.6e}",
        f"{stats['loss']['min']:.6e}",
        f"{stats['loss']['max']:.6e}",
        f"{stats['loss']['count']}",
    )
    # add SNR row
    table.add_row(
        "SNR (dB)",
        f"{stats['snr']['mean']:.6f}",
        f"{stats['snr']['std']:.6f}",
        f"{stats['snr']['min']:.6f}",
        f"{stats['snr']['max']:.6f}",
        f"{stats['snr']['count']}",
    )

    console.print(table)


def display_sample_details(
    console: Console,
    sample_details: List[Dict],
    unnormalize: bool,
    min_mag: float,
    max_mag: float,
    norm_eps: float,
) -> None:
    """Display sample details in a rich format, optionally un-normalizing."""
    if not sample_details:
        console.print(
            Panel(
                "[yellow]No samples were collected (evaluation might have failed early or num_samples_to_log=0).[/yellow]",
                title="Sample Predictions",
            )
        )
        return

    console.print(
        f"\n[bold cyan]Sample Predictions (Top {len(sample_details)})[/bold cyan]"
    )

    for sample in sample_details:
        table = Table(box=None, show_header=False)
        table.add_column("Property", style="blue", width=20)
        table.add_column("Value")

        table.add_row("Rx Position", str(sample["rx_pos"]))
        table.add_row("MSE Loss (Norm)", f"{sample['loss']:.6e}")
        table.add_row("SNR (dB)", f"{sample['snr']:.6f}")

        panel_content = table
        panel = Panel(
            panel_content,
            title=f"[bold]Sample Index {sample['index']}[/bold]",  # use original index
            border_style="green",
        )
        console.print(panel)

        h_gt_norm = sample["h_mag_gt"]
        h_pred_norm = sample["h_mag_pred"]

        # un-normalize if requested
        if unnormalize:
            scale = max_mag - min_mag
            # handle case where scale is zero or near zero
            if scale < norm_eps:
                h_gt_unnorm = torch.full_like(h_gt_norm, (max_mag + min_mag) / 2)
                h_pred_unnorm = torch.full_like(h_pred_norm, (max_mag + min_mag) / 2)
                unnorm_label = "(Constant)"
            else:
                h_gt_unnorm = h_gt_norm * scale + min_mag
                h_pred_unnorm = h_pred_norm * scale + min_mag
                unnorm_label = "(Un-normalized)"

            gt_title = Text(f"Ground Truth Magnitude {unnorm_label}", style="cyan")
            pred_title = Text(f"Predicted Magnitude {unnorm_label}", style="cyan")
            console.print(gt_title)
            console.print(format_magnitude_tensor(h_gt_unnorm))
            console.print(pred_title)
            console.print(format_magnitude_tensor(h_pred_unnorm))

        else:
            gt_title = Text("Ground Truth Magnitude (Normalized)", style="cyan")
            pred_title = Text("Predicted Magnitude (Normalized)", style="cyan")
            console.print(gt_title)
            console.print(format_magnitude_tensor(h_gt_norm))
            console.print(pred_title)
            console.print(format_magnitude_tensor(h_pred_norm))

        console.print("")  # add spacing


def evaluate(args):
    """Main evaluation function for magnitude prediction."""
    set_random_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    checkpoint_path = Path(args.checkpoint)
    log_dir = Path(args.log_dir)
    checkpoint_name = checkpoint_path.stem
    logger, console = setup_eval_logging(log_dir, checkpoint_name)

    console.rule("[bold blue]Magnitude Prediction Evaluation Start[/bold blue]")
    logger.info(f"Evaluation Arguments: {vars(args)}")
    logger.info(f"Using device: {device}")
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    # --- Load Model ---
    try:
        # load model state, don't need training_args for eval
        model_state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        model_config = model_state["config"]
        # load model using class method, specifying device
        model, load_iter = GaussianChannelFieldModel.load(
            checkpoint_path, device=device, training_args=None
        )
        model.eval()  # set model to evaluation mode
        logger.info(
            f"[green]Model loaded successfully from iteration {load_iter}.[/green]"
        )
        logger.info(f"Model Configuration: {model_config}")
        logger.info(f"Total Gaussians in loaded model: {model.get_xyz.shape[0]:,}")
    except Exception as e:
        logger.exception(f"[bold red]Failed to load checkpoint:[/bold red] {e}")
        return

    # --- Load Data ---
    logger.info(f"Loading data from: {args.data_path}")
    try:
        # get the validation dataloader
        val_loader = get_wireless_dataloader(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,  # no shuffle for evaluation
            train=False,  # use test split
            train_ratio=args.train_ratio,  # ensure correct split
            seed=args.seed,
            drop_last=False,  # keep all samples
            pin_memory=True,
            norm_eps=args.norm_eps,  # pass norm eps
        )
        # get metadata, including normalization parameters
        metadata = val_loader.dataset.get_metadata()
        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]
        # wavelength = metadata["wavelength"] # not needed for magnitude
        tx_position = metadata["tx_position"].to(device)
        min_mag = metadata.get("min_magnitude", 0.0)
        max_mag = metadata.get("max_magnitude", 1.0)

        # display metadata
        metadata_table = Table(title="Dataset Metadata")
        metadata_table.add_column("Property", style="cyan")
        metadata_table.add_column("Value")
        metadata_table.add_row("Dataset Path", args.data_path)
        metadata_table.add_row("Transmit Antennas (Nt)", str(nt))
        metadata_table.add_row("Receive Antennas (Nr)", str(nr))
        metadata_table.add_row("Frequency", f"{metadata['frequency']/1e9:.2f} GHz")
        # metadata_table.add_row("Wavelength", f"{wavelength:.4f} m")
        metadata_table.add_row("Is SISO", str(metadata["is_siso"]))
        metadata_table.add_row("Tx Position", str(tx_position.cpu().numpy()))
        metadata_table.add_row("Min Magnitude (Raw)", f"{min_mag:.4e}")
        metadata_table.add_row("Max Magnitude (Raw)", f"{max_mag:.4e}")
        metadata_table.add_row("Test Set Size", f"{len(val_loader.dataset)}")
        console.print(metadata_table)

    except Exception as e:
        logger.exception(f"[bold red]Failed to load data:[/bold red] {e}")
        return

    # --- Evaluation Loop ---
    criterion = nn.MSELoss().to(device)  # use MSE loss
    all_losses = []
    all_snrs = []
    sample_details: List[Dict] = []  # store detailed info for a few samples

    start_time = time.time()
    with torch.no_grad():  # disable gradients during evaluation
        # setup progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,  # use the rich console
        ) as progress:
            eval_task = progress.add_task("[cyan]Evaluating...", total=len(val_loader))

            for i, batch in enumerate(val_loader):
                rx_pos_batch = batch["rx_position"].to(device)
                # target is normalized magnitude
                h_mag_gt_batch = batch["channel_magnitude"].to(device)
                original_indices = batch["index"]  # get original indices
                current_batch_size = rx_pos_batch.shape[0]

                # render predicted magnitude
                h_mag_pred_batch = render_magnitude(
                    rx_positions=rx_pos_batch,
                    model=model,
                    tx_position=tx_position,
                    # wavelength=wavelength, # not needed
                    nt=nt,
                    nr=nr,
                    eps=args.snr_calc_eps,  # use eps for stability
                )

                # iterate through batch items for individual loss/snr and sample logging
                for j in range(current_batch_size):
                    h_mag_pred = h_mag_pred_batch[j].unsqueeze(
                        0
                    )  # add batch dim back for loss
                    h_mag_gt = h_mag_gt_batch[j].unsqueeze(0)

                    # calculate MSE loss for this sample
                    loss = criterion(h_mag_pred, h_mag_gt).item()  # get scalar value
                    # calculate SNR based on MSE and target magnitude
                    snr_tensor = calculate_snr(
                        torch.tensor(loss, device=device),
                        h_mag_gt,
                        eps=args.snr_calc_eps,
                    )
                    snr = snr_tensor.item()

                    # store metrics if they are valid numbers
                    if not np.isnan(loss) and not np.isinf(loss):
                        all_losses.append(loss)
                    # store SNR if valid
                    if not np.isnan(snr) and not np.isinf(snr):
                        all_snrs.append(snr)

                    # store details for the first few samples if requested
                    if len(sample_details) < args.num_samples_to_log:
                        sample_details.append(
                            {
                                "index": original_indices[j],  # store original index
                                "rx_pos": rx_pos_batch[j].cpu().numpy(),
                                "h_mag_gt": h_mag_gt_batch[j].cpu(),  # store on cpu
                                "h_mag_pred": h_mag_pred_batch[j].cpu(),  # store on cpu
                                "loss": loss,
                                "snr": snr,
                            }
                        )

                # update progress bar
                progress.update(eval_task, advance=1)

    end_time = time.time()
    eval_duration = end_time - start_time

    # --- Aggregate and Display Results ---
    losses_np = np.array(all_losses)
    snrs_np = np.array(all_snrs)

    # calculate statistics safely
    stats = {}
    if len(losses_np) > 0:
        stats["loss"] = {
            "mean": np.mean(losses_np),
            "std": np.std(losses_np),
            "min": np.min(losses_np),
            "max": np.max(losses_np),
            "count": len(losses_np),
        }
    else:  # handle empty results
        stats["loss"] = {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "count": 0,
        }

    if len(snrs_np) > 0:
        stats["snr"] = {
            "mean": np.mean(snrs_np),
            "std": np.std(snrs_np),
            "min": np.min(snrs_np),
            "max": np.max(snrs_np),
            "count": len(snrs_np),
        }
    else:  # handle empty results
        stats["snr"] = {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "count": 0,
        }

    console.rule("[bold blue]Evaluation Results[/bold blue]")
    console.print(
        f"[green]Evaluation completed in {eval_duration:.2f} seconds.[/green]"
    )
    console.print(f"Total samples evaluated: {stats['loss']['count']}")

    # display overall metrics
    overall_table = Table(box=None, show_header=False)
    overall_table.add_column("Metric", style="cyan", width=25)
    overall_table.add_column("Value", style="green")
    overall_table.add_row("Average MSE Loss (Norm)", f"{stats['loss']['mean']:.6e}")
    overall_table.add_row("Average SNR (dB)", f"{stats['snr']['mean']:.6f}")
    console.print(Panel(overall_table, title="[bold]Overall Metrics[/bold]"))

    # display detailed statistics table
    display_stats_table(console, stats)

    # display sample details if requested
    if args.num_samples_to_log > 0:
        display_sample_details(
            console,
            sample_details,
            args.unnormalize_samples,
            min_mag,
            max_mag,
            args.norm_eps,
        )

    console.rule("[bold blue]Evaluation End[/bold blue]")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
```

## File: datasets/dataloader.py
```python
# datasets/dataloader.py

from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# import the updated dataset class
from .wireless_dataset import WirelessDataset


def collate_wireless_batch(batch: list) -> Dict[str, Any]:
    """Collate function for wireless dataset batches.

    Args:
        batch (list): List of dataset items ({'rx_position': tensor, 'channel_magnitude': tensor, 'index': int}).

    Returns:
        Collated batch with stacked tensors and list of indices.
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        # stack tensors
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([item[key] for item in batch])
        # collect indices or other non-tensor data as lists
        elif isinstance(batch[0][key], (int, float, str)):
            collated[key] = [item[key] for item in batch]
        else:
            # fallback for other types, collect as list
            collated[key] = [item[key] for item in batch]

    return collated


def get_wireless_dataloader(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    train: bool = True,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
    drop_last: bool = False,
    pin_memory: bool = True,
    norm_eps: float = 1e-8,  # pass normalization epsilon
) -> DataLoader:
    """Create a DataLoader for the wireless magnitude dataset.

    Args:
        data_path (str): Path to the dataset file.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of workers for data loading.
        shuffle (bool): Whether to shuffle the data (typically True for train).
        train (bool): Whether to load training or test set.
        train_ratio (float): Ratio of data to use for training.
        seed (int, optional): Random seed for train/test split.
        drop_last (bool): Whether to drop the last incomplete batch (typically True for train).
        pin_memory (bool): Whether to use pinned memory for faster GPU transfer.
        norm_eps (float): Epsilon for dataset normalization stability.

    Returns:
        The configured data loader.
    """
    dataset = WirelessDataset(
        data_path,
        train=train,
        train_ratio=train_ratio,
        seed=seed,
        norm_eps=norm_eps,
    )

    # handle case where dataset split might be empty
    if len(dataset) == 0:
        print(
            f"Warning: DataLoader created for an empty dataset ({'train' if train else 'test'} split)."
        )
        # return a dataloader that yields nothing, or handle as needed
        return DataLoader(dataset, batch_size=batch_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_wireless_batch,
        pin_memory=pin_memory,
        drop_last=drop_last,
        # persistent_workers=True if num_workers > 0 else False # consider adding for efficiency
    )


def get_dataloaders(
    data_path: str,
    batch_size: int = 1,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
    pin_memory: bool = True,
    norm_eps: float = 1e-8,  # pass normalization epsilon
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Create training and validation DataLoaders for the wireless magnitude dataset.

    Args:
        data_path (str): Path to the dataset file.
        batch_size (int): Number of samples per batch for training loader.
        num_workers (int): Number of workers for data loading.
        train_ratio (float): Ratio of data to use for training.
        seed (int, optional): Random seed for train/test split.
        pin_memory (bool): Use pinned memory.
        norm_eps (float): Epsilon for dataset normalization stability.

    Returns:
        A tuple of (training loader, validation loader, dataset metadata).
        Validation loader always uses batch_size=1 and shuffle=False.
    """
    train_loader = get_wireless_dataloader(
        data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,  # shuffle training data
        train=True,
        train_ratio=train_ratio,
        seed=seed,
        drop_last=True,  # drop last incomplete batch for training
        pin_memory=pin_memory,
        norm_eps=norm_eps,
    )

    val_loader = get_wireless_dataloader(
        data_path,
        batch_size=1,  # typically use batch size 1 for validation
        num_workers=num_workers,
        shuffle=False,  # no need to shuffle validation data
        train=False,
        train_ratio=train_ratio,
        seed=seed,
        drop_last=False,  # keep all validation samples
        pin_memory=pin_memory,
        norm_eps=norm_eps,
    )

    # get metadata from one of the datasets (they share the same base data)
    # ensure train_loader.dataset exists even if empty
    metadata = {}
    if hasattr(train_loader, "dataset") and train_loader.dataset is not None:
        metadata = train_loader.dataset.get_metadata()
    elif hasattr(val_loader, "dataset") and val_loader.dataset is not None:
        metadata = val_loader.dataset.get_metadata()
        print(
            "Warning: Using metadata from validation dataset as training dataset might be empty."
        )
    else:
        print(
            "Warning: Could not retrieve metadata as both train and val datasets seem unavailable."
        )
        # provide default or raise error depending on requirements
        metadata = {  # provide some defaults maybe?
            "num_tx_ant": 0,
            "num_rx_ant": 0,
            "frequency": 0,
            "wavelength": 0,
            "is_siso": False,
            "tx_position": torch.zeros(3),
            "env_dims": None,
            "point_cloud": None,
            "min_magnitude": 0,
            "max_magnitude": 1,
            "norm_eps": norm_eps,
        }

    return train_loader, val_loader, metadata
```

## File: datasets/indoor.m
```
%% environment setup
close all force; clear; clc;
plot_rays = false;
visualize = false;

mapFileName = "models/conference.stl";
[stl_data, ~] = stlread(mapFileName);

if visualize
    viewer = siteviewer("SceneModel", mapFileName, "Transparency", 0.25);
end

% environment dimensions setup
vertices = stl_data.Points;
faces = stl_data.ConnectivityList;

xy_offset = 0.1;
z_offset = 0.1;
min_z = max(0, min(vertices(:, 3)));
env_dims = [
            [min(vertices(:, 1)) + xy_offset, max(vertices(:, 1)) - xy_offset];
            [min(vertices(:, 2)) + xy_offset, max(vertices(:, 2)) - xy_offset];
            [min_z, max(vertices(:, 3)) - z_offset]
            ];

% point cloud generation params
pc_params = struct();
pc_params.edge_density = 2.1;
pc_params.surface_density = 1.6;
pc_params.volume_density = 0;
pc_params.boundary_density = 87.3;
pc_params.random_points = 0;
pc_params.noise_std = 0;
pc_params.edge_reduction = 1;
pc_params.surface_reduction = 1;

%% generate point cloud
point_cloud = generate_pc(vertices, faces, pc_params, env_dims, visualize);

%% system config
fc = 5e9;
lambda = physconst("lightspeed") / fc;

% OFDM parameters
cfg = wlanNonHTConfig;
cfg.ChannelBandwidth = 'CBW80';

% extra config
use_single_sc = true;
sc_idx = [];
use_siso = true;

if use_siso
    % single-input single-output
    txArray = phased.IsotropicAntennaElement();
    rxArray = phased.IsotropicAntennaElement();

    num_tx_ant = 1;
    num_rx_ant = 1;
else
    % multiple-input multiple-output
    txArray = phased.URA("Size", [4 4], "ElementSpacing", lambda / 2);
    rxArray = phased.ULA("NumElements", 2, "ElementSpacing", lambda / 2);

    num_tx_ant = prod(txArray.Size);
    num_rx_ant = rxArray.NumElements;
end

%% AP setup
AP = txsite("cartesian", ...
    "Antenna", txArray, ...
    "AntennaPosition", [-1.5; 0.0; 2.1], ... % Positioned near ceiling
    "TransmitterFrequency", fc, ...
    "TransmitterPower", 0.05);

%% user setup
approx_target_users = 626;

% seed
S = RandStream("mt19937ar", "Seed", 17);
RandStream.setGlobalStream(S);

user_params = struct();
user_params.check_building_collision = false;
user_params.check_user_collision = true;
[Users, actual_users] = create_users(env_dims, approx_target_users, rxArray, user_params, []);

if actual_users < approx_target_users
    error('Failed to create all requested users. Only created %d out of %d users.', actual_users, approx_target_users);
end

%% RT simulation
method = "sbr"; % "image" | "sbr"
max_refs = 1;

pm = propagationModel("raytracing", ...
    "Method", method, ...
    "CoordinateSystem", "cartesian", ...
    "SurfaceMaterial", "wood", ...
    "TerrainMaterial", "wood", ...
    "MaxNumReflections", max_refs, ...
    "UseGPU", "auto");

rays = raytrace(AP, Users, pm, "Map", mapFileName);

% filter users to keep only those with valid rays
valid_user_mask = ~cellfun(@isempty, rays);
Users = Users(valid_user_mask);
rays = rays(valid_user_mask);
num_users = sum(valid_user_mask);

if num_users < approx_target_users
    warning('Only %d out of %d users had valid rays, discarding the rest.', ...
        num_users, approx_target_users);
end

%% visualize
if visualize
    show(AP, "ShowAntennaHeight", false)
    show(Users, "ShowAntennaHeight", false)

    if plot_rays

        for userIdx = 1:(num_users / 20) % ~20 % rays
            plot(rays{userIdx}, "Colormap", jet)
            pause(0.05)
        end

    end

end

%% extract positions
rx_positions = zeros(3, num_users);

for userIdx = 1:num_users
    rx_positions(:, userIdx) = Users(userIdx).AntennaPosition;
end

%% CSI collection
ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
activeIndices = ofdmInfo.ActiveFrequencyIndices;

if use_single_sc

    if isempty(sc_idx)
        sc_idx = ceil(length(activeIndices) / 2);
    end

    numSubcarriers = 1;
    sc_spacing = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;
    freqs = fc + activeIndices(sc_idx) * sc_spacing;
else
    numSubcarriers = length(activeIndices);
    sc_spacing = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;
    freqs = fc + activeIndices * sc_spacing;
end

H = zeros(num_users, num_tx_ant, num_rx_ant, numSubcarriers);
AoD_all = cell(num_users, 1);
AoA_all = cell(num_users, 1);
path_loss = zeros(num_users, 1);
path_loss_per_ray = cell(num_users, 1);

for userIdx = 1:num_users
    [H(userIdx, :, :, :), AoD_all{userIdx}, AoA_all{userIdx}] = ...
        generate_csi(rays{userIdx}, fc, cfg, txArray, rxArray, method, 'indoor', use_single_sc, sc_idx);
    path_loss(userIdx) = mean([rays{userIdx}.PathLoss]);
    path_loss_per_ray{userIdx} = [rays{userIdx}.PathLoss];
end

% check for null values in channel matrix
if any(isnan(H(:)))
    warning('Channel matrix contains NaN values!');
end

%% ray marching and per-ray information (for NeWRF comparison)
ray_steps = cell(num_users, 1);
ray_points = cell(num_users, 1);
ray_interactions = cell(num_users, 1);
ray_coefficients = cell(num_users, 1);

for userIdx = 1:num_users
    [ray_steps{userIdx}, ray_points{userIdx}] = ray_marching(rays{userIdx});
    [ray_interactions{userIdx}, ray_coefficients{userIdx}] = get_ray_chan(rays{userIdx}, freqs, method);
end

% check for null values in ray marching results
if any(cellfun(@(x) any(cellfun(@(y) any(isnan(y(:))), x)), ray_steps)) || ...
        any(cellfun(@(x) any(cellfun(@(y) any(isnan(y(:))), x)), ray_points)) || ...
        any(cellfun(@(x) any(isnan(x(:))), ray_coefficients))
    warning('Ray marching results contain NaN values!');
end

%% save
output_dir = "outputs";
mkdir(output_dir);

[~, mapname] = fileparts(mapFileName);

% create dataset
dataset = struct();

dataset.config.tx_antennas = num_tx_ant;
dataset.config.rx_antennas = num_rx_ant;
dataset.config.frequency = fc;
dataset.config.wavelength = lambda;
dataset.config.num_users = num_users;
dataset.config.use_siso = use_siso;

dataset.environment.dimensions = env_dims;
dataset.environment.point_cloud = point_cloud;
dataset.environment.pc_params = pc_params;

dataset.nodes.ap_position = AP.AntennaPosition';
dataset.nodes.users_positions = rx_positions;

dataset.channel.H = H;
% NOTE: AoD is unneeded as it is primarily related to the txsite
% dataset.channel.AoD = AoD_all;
dataset.channel.path_loss = path_loss;

% truncate data (max 10 paths)
[AoA_trunc, AoD_trunc, path_loss_per_ray_trunc] = truncate_data(AoA_all, AoD_all, path_loss_per_ray, 10);
dataset.channel.AoA = AoA_trunc;
dataset.channel.AoD = AoD_trunc;
dataset.channel.path_loss_per_ray = path_loss_per_ray_trunc;

dataset.channel.ray_steps = ray_steps;
dataset.channel.ray_points = ray_points;
dataset.channel.ray_interactions = ray_interactions;
dataset.channel.ray_coefficients = ray_coefficients;
dataset.channel.frequencies = freqs;

sc_str = '';

if use_single_sc
    sc_str = sprintf('_sc%d', sc_idx);
end

filename = sprintf('%s/conf_%dx%d_%du_%.1fghz_%sRT%s.mat', ...
    output_dir, ...
    num_tx_ant, ...
    num_rx_ant, ...
    num_users, ...
    fc / 1e9, ...
    method, ...
    sc_str);

save(filename, 'dataset', '-v7.3');

function [AoA_trunc, AoD_trunc, path_loss_trunc] = truncate_data(AoA, AoD, path_loss, max_paths)
    num_users = length(AoA);
    AoA_trunc = cell(num_users, 1);
    AoD_trunc = cell(num_users, 1);
    path_loss_trunc = cell(num_users, 1);
    
    for i = 1:num_users
        if ~isempty(AoA{i})
            % sort paths by path loss
            [sorted_pl, sort_idx] = sort(path_loss{i});
            sorted_AoA = AoA{i}(:, sort_idx);
            sorted_AoD = AoD{i}(:, sort_idx);
            
            % take top max_paths with lowest path loss
            num_paths = min(length(sorted_pl), max_paths);
            AoA_trunc{i} = sorted_AoA(:, 1:num_paths);
            AoD_trunc{i} = sorted_AoD(:, 1:num_paths);
            path_loss_trunc{i} = sorted_pl(1:num_paths);
        else
            AoA_trunc{i} = [];
            AoD_trunc{i} = [];
            path_loss_trunc{i} = [];
        end
    end
end
```

## File: models/gaussian_model.py
```python
# models/gaussian_model.py

import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import inverse_sigmoid  # still potentially useful for init opacity logit
from utils import build_covariance_inverse, build_rotation, get_expon_lr_func

# import updated networks
from .networks import AttributeNetwork, ContributionDecoderNetwork


class GaussianChannelFieldModel(nn.Module):
    """Gaussian channel field (GCF) model for predicting channel magnitude."""

    def __init__(
        self,
        num_tx_ant: int,
        num_rx_ant: int,
        latent_dim: int,
        attribute_hidden_dim: int = 64,
        attribute_num_layers: int = 3,
        attribute_pos_enc_freqs: int = 10,
        decoder_hidden_dim: int = 64,
        decoder_num_layers: int = 4,  # increased default slightly
        initial_gaussians: int = 30000,
        init_opacity_value: float = 0.1,  # initial base activation (before sigmoid) related value
        init_scale_value: float = 0.02,  # initial scale (before exp) related value
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.latent_dim = latent_dim
        self.device = device

        # gaussian parameters (initialized later)
        self._xyz = nn.Parameter(torch.empty(0, 3, device=device))
        self._rotation = nn.Parameter(
            torch.empty(0, 4, device=device)
        )  # quaternion (w, x, y, z)
        self._scaling = nn.Parameter(torch.empty(0, 3, device=device))  # log scale

        # networks
        self.attribute_network = AttributeNetwork(
            latent_dim=latent_dim,
            mlp_hidden_dim=attribute_hidden_dim,
            mlp_num_layers=attribute_num_layers,
            pos_encoding_freqs=attribute_pos_enc_freqs,
        ).to(device)

        # decoder predicts normalized magnitude contributions (Nt * Nr outputs)
        self.contribution_decoder = ContributionDecoderNetwork(
            latent_dim=latent_dim,
            output_dim=num_tx_ant * num_rx_ant,  # direct magnitude prediction
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            # sigmoid activation is now inside ContributionDecoderNetwork
        ).to(device)

        # initialization helpers
        self.initial_gaussians = initial_gaussians
        # store the logit corresponding to the initial opacity value
        self.init_opacity_logit = inverse_sigmoid(
            torch.tensor(init_opacity_value, device=device)
        )
        # store the log scale corresponding to the initial scale value
        self.init_log_scale = torch.log(torch.tensor(init_scale_value, device=device))

        self.optimizer = None
        self.lr_schedulers = {}

        self.setup_activations()

    def setup_activations(self):
        """Setup activation functions for Gaussian parameters."""
        self.scaling_activation = torch.exp  # scales are stored in log space
        self.opacity_activation = torch.sigmoid  # activates the base activation logits
        self.rotation_activation = lambda r: F.normalize(
            r, p=2, dim=-1
        )  # normalize quaternions

    @property
    def get_xyz(self):
        """Returns Gaussian positions (means)."""
        return self._xyz

    @property
    def get_scaling(self):
        """Returns activated and clamped Gaussian scales."""
        # clamp to prevent scales from becoming too small or zero
        return self.scaling_activation(self._scaling).clamp(min=1e-8)

    @property
    def get_rotation(self):
        """Returns normalized Gaussian rotations (quaternions)."""
        return self.rotation_activation(self._rotation)

    def get_attributes(
        self, tx_position: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes latent features and base activation logits dynamically."""
        if self._xyz.shape[0] == 0:
            # handle case with no gaussians
            return torch.empty(0, self.latent_dim, device=self.device), torch.empty(
                0, 1, device=self.device
            )

        # ensure tx_position is expanded correctly
        if tx_position.dim() == 1:
            tx_position_exp = tx_position.unsqueeze(0).expand(self._xyz.shape[0], -1)
        elif tx_position.shape[0] == 1:
            tx_position_exp = tx_position.expand(self._xyz.shape[0], -1)
        elif tx_position.shape[0] == self._xyz.shape[0]:
            tx_position_exp = tx_position
        else:
            raise ValueError("tx_position shape mismatch")

        # pass means and expanded tx position to attribute network
        latent_features, base_activations_logits = self.attribute_network(
            self._xyz, tx_position_exp
        )
        return latent_features, base_activations_logits  # return logits directly

    def get_opacity_activated(self, tx_position: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid-activated base activations, computed dynamically."""
        _, base_activations_logits = self.get_attributes(tx_position)
        # apply sigmoid activation to the logits
        return self.opacity_activation(base_activations_logits)

    def get_covariance(
        self, return_inverse=False, eps=1e-6
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Computes covariance matrix Σ and optionally its inverse Σ^-1."""
        scaling = self.get_scaling  # activated scales
        rotation_q = self.get_rotation  # normalized quaternions

        if scaling.shape[0] == 0:
            # handle no gaussians case
            empty_cov = torch.empty(0, 3, 3, device=self.device)
            return (empty_cov, empty_cov) if return_inverse else empty_cov

        # build rotation matrix from quaternion
        R = build_rotation(rotation_q)  # (N, 3, 3)
        # create diagonal matrix for squared scales
        S_sq_diag = torch.diag_embed(scaling * scaling)  # (N, 3, 3)
        # compute covariance: Sigma = R * S^2 * R^T
        covariance = R @ S_sq_diag @ R.transpose(1, 2)

        if return_inverse:
            # compute inverse covariance: Sigma^-1 = R * S^-2 * R^T
            inv_covariance = build_covariance_inverse(R, scaling, eps)

            # check for NaNs/Infs in inverse covariance (can happen if scale is near zero)
            if torch.isnan(inv_covariance).any() or torch.isinf(inv_covariance).any():
                print(
                    "Warning: NaN or Inf detected in inverse covariance. Replacing offending matrices with identity."
                )
                # identify gaussians with bad inverse covariance
                bad_indices = torch.isnan(inv_covariance).any(dim=(1, 2)) | torch.isinf(
                    inv_covariance
                ).any(dim=(1, 2))
                # create identity matrices for replacement
                identity = torch.eye(
                    3, device=self.device, dtype=inv_covariance.dtype
                ).expand(bad_indices.sum(), -1, -1)
                # replace bad matrices
                inv_covariance[bad_indices] = identity
            return covariance, inv_covariance
        else:
            return covariance

    def init_gaussians(
        self,
        env_dims: Optional[torch.Tensor] = None,
        num_points: Optional[int] = None,
        point_cloud: Optional[torch.Tensor] = None,
    ):
        """Initializes Gaussian parameters (position, rotation, scale)."""
        num_to_init = num_points if num_points is not None else self.initial_gaussians
        if num_to_init <= 0:
            print("Warning: No Gaussians requested for initialization.")
            self._xyz = nn.Parameter(
                torch.empty(0, 3, device=self.device).requires_grad_(True)
            )
            self._rotation = nn.Parameter(
                torch.empty(0, 4, device=self.device).requires_grad_(True)
            )
            self._scaling = nn.Parameter(
                torch.empty(0, 3, device=self.device).requires_grad_(True)
            )
            return

        # initialize positions (xyz)
        if point_cloud is not None:
            num_available_points = point_cloud.shape[0]
            print(f"Point cloud provided with {num_available_points} points.")
            if num_available_points == 0:
                print(
                    "Warning: Point cloud is empty. Falling back to random initialization within env_dims (if provided)."
                )
                point_cloud = None  # fallback
            else:
                if num_to_init > num_available_points:
                    print(
                        f"Warning: Requested {num_to_init} Gaussians, but point cloud only has {num_available_points}. "
                        f"Using all {num_available_points} points and potentially adding random points."
                    )
                    # use all points and maybe add more later if needed? or just use available? let's use available.
                    num_to_init = num_available_points
                    indices = torch.arange(num_available_points)

                else:
                    print(
                        f"Randomly sampling {num_to_init} points from the point cloud."
                    )
                    # sample without replacement
                    indices_np = np.random.choice(
                        num_available_points, num_to_init, replace=False
                    )
                    indices = torch.from_numpy(indices_np).long()

                # select points and ensure correct type/device
                xyz = point_cloud[indices].to(self.device).float()
                if xyz.shape[1] != 3:
                    raise ValueError(
                        f"Point cloud must have shape (N, 3), got {point_cloud.shape}"
                    )

        # if no point cloud or fallback needed
        if point_cloud is None:
            if env_dims is not None and env_dims.shape == (3, 2):
                print(
                    f"Initializing {num_to_init} random Gaussians within environment dimensions."
                )
                env_min = env_dims[:, 0].to(self.device)
                env_max = env_dims[:, 1].to(self.device)
                if env_min.shape != (3,) or env_max.shape != (3,):
                    raise ValueError(
                        f"env_dims should result in shapes (3,), got min: {env_min.shape}, max: {env_max.shape}"
                    )
                # generate random points within bounds
                xyz = (
                    torch.rand(num_to_init, 3, device=self.device) * (env_max - env_min)
                    + env_min
                )
            else:
                # default random initialization if no bounds given
                print(
                    f"Warning: No point cloud or valid env_dims provided. "
                    f"Initializing {num_to_init} random Gaussians in [-1, 1] range."
                )
                xyz = (
                    torch.rand(num_to_init, 3, device=self.device) * 2 - 1
                ) * 1.0  # scale appropriately if needed

        # set parameters
        self._xyz = nn.Parameter(xyz.requires_grad_(True))

        # initialize scales (log scale)
        scales = torch.full(
            (num_to_init, 3), self.init_log_scale.item(), device=self.device
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))

        # initialize rotations (identity quaternion: w=1, x=y=z=0)
        rots = torch.zeros((num_to_init, 4), device=self.device)
        rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots.requires_grad_(True))

        print(f"GCF Model initialized with {self.get_xyz.shape[0]} Gaussians.")

    def get_params(self, lr_dict: Dict[str, float]) -> list:
        """Returns parameter groups for the optimizer with specified learning rates."""
        param_groups = [
            {"params": [self._xyz], "lr": lr_dict.get("xyz", 0.0), "name": "xyz"},
            {
                "params": [self._rotation],
                "lr": lr_dict.get("rotation", 0.0),
                "name": "rotation",
            },
            {
                "params": [self._scaling],
                "lr": lr_dict.get("scaling", 0.0),
                "name": "scaling",
            },
            # include network parameters
            {
                "params": self.attribute_network.parameters(),
                "lr": lr_dict.get("attribute_net", 0.0),
                "name": "attribute_net",
            },
            {
                "params": self.contribution_decoder.parameters(),
                "lr": lr_dict.get("decoder", 0.0),
                "name": "decoder",
            },
        ]
        # filter out groups with no parameters (can happen if networks are empty/frozen)
        # param_groups = [pg for pg in param_groups if len(pg['params']) > 0]
        return param_groups

    def training_setup(self, training_args: Any):
        """Setup optimizer (AdamW) and learning rate schedulers based on training args."""

        # map training args LRs to parameter group names
        lr_map = {
            "xyz": training_args.position_lr_init,
            "rotation": training_args.rotation_lr,
            "scaling": training_args.scaling_lr,
            "attribute_net": training_args.attribute_net_lr,
            "decoder": training_args.decoder_lr,
        }
        params = self.get_params(lr_map)

        # use AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            params,
            lr=0.0,  # initial LR set by scheduler
            eps=(
                training_args.optimizer_eps
                if hasattr(training_args, "optimizer_eps")
                else 1e-8
            ),  # allow configuring eps
            weight_decay=training_args.weight_decay,
        )
        print(
            f"Optimizer AdamW initialized with weight decay: {training_args.weight_decay}"
        )

        # setup LR schedulers
        self.lr_schedulers = {}
        # exponential decay for positions
        self.lr_schedulers["xyz"] = get_expon_lr_func(
            lr_init=training_args.position_lr_init,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.iterations,
        )

        # constant LR for other parameters (can be changed to schedulers if needed)
        for name, lr_init in lr_map.items():
            if name != "xyz":
                # use a simple lambda for constant LR
                self.lr_schedulers[name] = lambda step, lr=lr_init: lr
        print("Learning rate schedulers set up.")

    def update_learning_rate(self, iteration: int, training_args: Any):
        """Update learning rates for all parameter groups based on schedulers and iteration."""
        if not self.optimizer:
            print("Warning: Optimizer not initialized, cannot update learning rate.")
            return

        for param_group in self.optimizer.param_groups:
            name = param_group["name"]
            if name in self.lr_schedulers:
                # get the scheduled LR
                new_lr = self.lr_schedulers[name](iteration)

                # apply position freezing logic
                if name == "xyz" and iteration >= training_args.stop_xyz_iter:
                    new_lr = 0.0  # freeze position updates

                # assign the new LR to the parameter group
                param_group["lr"] = new_lr
            # else:
            # print(f"Warning: No LR scheduler found for parameter group '{name}'.")

    def save(self, filepath: Path, iteration: Optional[int] = None):
        """Save model state, optimizer state, and configuration."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        # ensure parameters are detached and on CPU for saving
        state_dict = {
            "iteration": iteration,
            "xyz": self._xyz.detach().cpu(),
            "rotation": self._rotation.detach().cpu(),
            "scaling": self._scaling.detach().cpu(),
            "attribute_network_state_dict": self.attribute_network.state_dict(),
            "decoder_state_dict": self.contribution_decoder.state_dict(),
            # save optimizer state if it exists
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else None
            ),
            # save model configuration
            "config": {
                "num_tx_ant": self.num_tx_ant,
                "num_rx_ant": self.num_rx_ant,
                "latent_dim": self.latent_dim,
                "attribute_hidden_dim": self.attribute_network.network.hidden_dim,
                "attribute_num_layers": self.attribute_network.network.num_layers,
                "attribute_pos_enc_freqs": self.attribute_network.pos_encoder_mean.num_freqs,
                "decoder_hidden_dim": self.contribution_decoder.hidden_dim,
                "decoder_num_layers": self.contribution_decoder.num_layers,
                # store initial values used, might be useful
                # "initial_gaussians": self.initial_gaussians,
                # "init_opacity_value": self.init_opacity_logit, # maybe save original value?
                # "init_scale_value": self.init_log_scale, # maybe save original value?
            },
        }
        torch.save(state_dict, str(filepath))
        # print(f"Model state saved to {filepath} at iteration {iteration}")

    @classmethod
    def load(
        cls, filepath: Path, device: torch.device, training_args: Optional[Any] = None
    ):
        """Load model state from a checkpoint."""
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        # load state dict onto the specified device
        state_dict = torch.load(
            str(filepath), map_location=device, weights_only=False
        )  # weights_only=False needed for optimizer
        config = state_dict["config"]

        # create a new model instance with the loaded configuration
        model = cls(
            num_tx_ant=config["num_tx_ant"],
            num_rx_ant=config["num_rx_ant"],
            latent_dim=config["latent_dim"],
            attribute_hidden_dim=config.get(
                "attribute_hidden_dim", 64
            ),  # use get for backward compatibility
            attribute_num_layers=config.get("attribute_num_layers", 3),
            attribute_pos_enc_freqs=config.get("attribute_pos_enc_freqs", 10),
            decoder_hidden_dim=config.get("decoder_hidden_dim", 64),
            decoder_num_layers=config.get("decoder_num_layers", 4),  # updated default
            device=device,
            # initial_gaussians etc. are not needed here as parameters are loaded directly
        )

        # load gaussian parameters
        model._xyz = nn.Parameter(state_dict["xyz"].to(device).requires_grad_(True))
        model._rotation = nn.Parameter(
            state_dict["rotation"].to(device).requires_grad_(True)
        )
        model._scaling = nn.Parameter(
            state_dict["scaling"].to(device).requires_grad_(True)
        )

        # load network states
        model.attribute_network.load_state_dict(
            state_dict["attribute_network_state_dict"]
        )
        model.contribution_decoder.load_state_dict(state_dict["decoder_state_dict"])

        # get iteration number from checkpoint
        iteration = state_dict.get("iteration", 0)  # default to 0 if not found

        # setup training components (optimizer, schedulers) if training_args are provided
        if training_args is not None:
            model.training_setup(
                training_args
            )  # re-initialize optimizer and schedulers
            # load optimizer state if available in checkpoint and optimizer exists
            if model.optimizer and state_dict.get("optimizer_state_dict"):
                try:
                    model.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                    # ensure optimizer state tensors are on the correct device
                    for state in model.optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                    print("Optimizer state loaded successfully.")
                except Exception as e:
                    print(
                        f"Warning: Could not load optimizer state: {e}. Optimizer state reset."
                    )
                    # reset optimizer state if loading fails
                    model.optimizer.state = {}  # or re-initialize?
            else:
                print(
                    "Optimizer state not found in checkpoint or optimizer not setup for loading."
                )
        else:
            print("No training_args provided, optimizer state not loaded.")

        print(f"GCF Model loaded from {filepath} (iteration {iteration}).")
        print(f"Loaded model has {model.get_xyz.shape[0]} Gaussians.")

        return model, iteration
```

## File: train.py
```python
# train.py

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.dataloader import get_dataloaders

# import the updated rendering engine
from engine.render_magnitude import render_magnitude
from models.gaussian_model import GaussianChannelFieldModel
from utils.general_utils import set_random_seed

# import updated loss and snr calculation
from utils.loss import calculate_snr  # MSELoss is used directly from nn
from utils.train_utils import compute_grad_stats, setup_logging


def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train Gaussian Channel Field Model for Magnitude Prediction"
    )

    # --- Data and Initialization ---
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset file (.mat)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training",
    )
    parser.add_argument(
        "--initial_gaussians",
        type=int,
        default=2_000,
        help="Number of Gaussians to initialize",
    )
    parser.add_argument(
        "--init_method",
        type=str,
        default="random",
        choices=["random", "point_cloud"],
        help="Initialization method ('random', 'point_cloud')",
    )
    parser.add_argument(
        "--norm_eps",
        type=float,
        default=1e-8,
        help="Epsilon for dataset magnitude normalization stability",
    )

    # --- Model Architecture ---
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Dimension of Gaussian latent features (F)",
    )
    parser.add_argument(
        "--attribute_hidden_dim",
        type=int,
        default=128,
        help="Hidden dim for Attribute Network MLP",
    )
    parser.add_argument(
        "--attribute_num_layers",
        type=int,
        default=4,
        help="Number of layers for Attribute Network MLP",
    )
    parser.add_argument(
        "--attribute_pos_enc_freqs",
        type=int,
        default=42,
        help="Num frequencies for positional encoding in Attribute Net",
    )
    parser.add_argument(
        "--decoder_hidden_dim",
        type=int,
        default=32,
        help="Hidden dim for Contribution Decoder MLP",
    )
    parser.add_argument(
        "--decoder_num_layers",
        type=int,
        default=2,
        help="Number of layers for Contribution Decoder MLP",
    )

    # --- Training ---
    parser.add_argument(
        "--iterations",
        type=int,
        default=30_000,
        help="Total training iterations (increased default)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (increased default)",
    )
    parser.add_argument(
        "--optimizer_eps",
        type=float,
        default=1e-8,
        help="AdamW optimizer epsilon",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay for AdamW optimizer (small default)",
    )  # added small default wd
    # parser.add_argument("--loss_eps", type=float, default=1e-10, help="Epsilon for SNR calculation stability (default: 1e-10)") # Renamed from loss_eps
    parser.add_argument(
        "--snr_calc_eps",
        type=float,
        default=1e-10,
        help="Epsilon for SNR calculation stability",
    )
    parser.add_argument(
        "--rx_noise_std",
        type=float,
        default=0.02,
        help="Std dev of Gaussian noise added to Rx positions during training (small default)",
    )  # added small default noise
    parser.add_argument(
        "--lambda_activation_l1",
        type=float,
        default=1e-2,
        help="L1 regularization weight for base activations logits (small default)",
    )  # added small default reg

    # --- Learning Rates ---
    parser.add_argument(
        "--position_lr_init",
        type=float,
        default=1e-4,
        help="Initial LR for Gaussian positions",
    )
    parser.add_argument(
        "--position_lr_final",
        type=float,
        default=1e-6,
        help="Final LR for Gaussian positions",
    )
    parser.add_argument(
        "--position_lr_delay_mult",
        type=float,
        default=0.01,
        help="Multiplier for position LR delay phase",
    )
    parser.add_argument(
        "--rotation_lr", type=float, default=0.001, help="LR for Gaussian rotations"
    )
    parser.add_argument(
        "--scaling_lr", type=float, default=0.005, help="LR for Gaussian scaling"
    )
    parser.add_argument(
        "--attribute_net_lr", type=float, default=0.001, help="LR for Attribute Network"
    )
    parser.add_argument(
        "--decoder_lr",
        type=float,
        default=0.0015,
        help="LR for the contribution decoder network (slightly increased)",
    )  # slightly increased decoder LR
    parser.add_argument(
        "--stop_xyz_iter",
        type=int,
        default=int(0.75 * 50_000),
        help="Stop updating Gaussian positions after this iteration",
    )  # adjusted default

    # --- Logging and Saving ---
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/magnitude_prediction",
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        default=20,
        help="Log training metrics every N iterations",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=250,
        help="Evaluate on validation set every N iterations",
    )  # less frequent eval
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=2000,
        help="Save checkpoint every N iterations",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )

    # --- System ---
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training",
    )

    args = parser.parse_args()

    # update stop_xyz_iter based on potentially changed iterations
    if hasattr(args, "iterations"):
        args.stop_xyz_iter = int(0.5 * args.iterations)
    else:
        # fallback if iterations isn't parsed correctly (shouldn't happen)
        args.stop_xyz_iter = float("inf")

    return args


def evaluate(
    model: GaussianChannelFieldModel,
    val_loader: DataLoader,
    criterion: nn.Module,  # expects MSELoss instance
    device: torch.device,
    tx_position: torch.Tensor,
    # wavelength: float, # wavelength not directly needed for magnitude rendering
    nt: int,
    nr: int,
    snr_calc_eps: float,
) -> Dict[str, float]:
    """Evaluates the model on the validation set for magnitude prediction."""
    model.eval()  # set model to evaluation mode
    total_loss = 0.0
    total_snr = 0.0
    count = 0

    with torch.no_grad():  # disable gradient calculations
        # wrap val_loader with tqdm for progress bar
        for batch in tqdm(
            val_loader, desc="Evaluating", leave=False, dynamic_ncols=True
        ):
            rx_pos_batch = batch["rx_position"].to(device)
            # target is now normalized magnitude
            h_mag_gt_batch = batch["channel_magnitude"].to(device)
            batch_size = rx_pos_batch.shape[0]

            # render predicted magnitude
            h_mag_pred_batch = render_magnitude(
                rx_positions=rx_pos_batch,
                model=model,
                tx_position=tx_position,
                # wavelength=wavelength, # not needed
                nt=nt,
                nr=nr,
                eps=snr_calc_eps,  # use snr_calc_eps for rendering stability too
            )

            # calculate MSE loss
            loss = criterion(h_mag_pred_batch, h_mag_gt_batch)
            # calculate SNR based on MSE and target magnitude
            snr = calculate_snr(loss, h_mag_gt_batch, eps=snr_calc_eps)

            total_loss += loss.item() * batch_size
            # accumulate SNR only if it's finite
            if not torch.isinf(snr) and not torch.isnan(snr):
                total_snr += snr.item() * batch_size
            # else:
            # print(f"Warning: Skipping Inf/NaN SNR value ({snr.item()}) in validation accumulation.")

            count += batch_size

    # calculate average loss and SNR
    avg_loss = total_loss / count if count > 0 else 0.0
    avg_snr = total_snr / count if count > 0 else float("-inf")  # or float('nan')?

    return {"val_mse_loss": avg_loss, "val_snr_db": avg_snr}


def train(args):
    """Main training loop for magnitude prediction."""
    set_random_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    # setup logging directory
    run_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_mag_L{args.latent_dim}_N{args.initial_gaussians}"
    )
    log_dir = Path(args.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = log_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # setup logging
    logger = setup_logging(log_dir)
    writer = SummaryWriter(str(log_dir / "tensorboard")) if args.tensorboard else None

    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Arguments: {vars(args)}")  # log all arguments
    logger.info(f"Using device: {device}")

    # --- Data Loading ---
    logger.info("Loading data...")
    try:
        train_loader, val_loader, metadata = get_dataloaders(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            seed=args.seed,
            norm_eps=args.norm_eps,  # pass norm eps
        )
        nt = metadata["num_tx_ant"]
        nr = metadata["num_rx_ant"]
        # wavelength = metadata["wavelength"] # not directly used in magnitude rendering
        tx_position = metadata["tx_position"].to(device)
        env_dims = metadata.get("env_dims")  # use get for safety
        point_cloud = metadata.get("point_cloud")
        min_mag = metadata.get("min_magnitude", 0.0)  # get normalization params
        max_mag = metadata.get("max_magnitude", 1.0)

        logger.info(
            f"Dataset Metadata: Nt={nt}, Nr={nr}, Freq={metadata['frequency']/1e9:.2f}GHz, IsSISO={metadata['is_siso']}"
        )
        logger.info(f"Magnitude Normalization: Min={min_mag:.4e}, Max={max_mag:.4e}")
        logger.info(f"Transmitter Position: {tx_position.cpu().numpy()}")
        if point_cloud is not None:
            logger.info(f"Point cloud loaded with shape: {point_cloud.shape}")
            point_cloud = point_cloud.to(
                device
            )  # move to device if using point cloud init
        else:
            logger.info("No point cloud data found or used for initialization.")
        if env_dims is not None:
            logger.info(f"Environment dimensions loaded: {env_dims.numpy().tolist()}")
            env_dims = env_dims.to(device)  # move to device if using random init
        else:
            logger.warning(
                "No environment dimensions found. Random init will use default range [-1, 1]."
            )

        # check if dataloaders are empty
        if len(train_loader) == 0:
            logger.error(
                "Training dataloader is empty! Check dataset path and train_ratio."
            )
            if writer:
                writer.close()
            return
        if len(val_loader) == 0:
            logger.warning(
                "Validation dataloader is empty! Evaluation will be skipped."
            )

    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        if writer:
            writer.close()
        return

    # --- Model Initialization ---
    logger.info("Initializing model...")
    model = GaussianChannelFieldModel(
        num_tx_ant=nt,
        num_rx_ant=nr,
        latent_dim=args.latent_dim,
        attribute_hidden_dim=args.attribute_hidden_dim,
        attribute_num_layers=args.attribute_num_layers,
        attribute_pos_enc_freqs=args.attribute_pos_enc_freqs,
        decoder_hidden_dim=args.decoder_hidden_dim,
        decoder_num_layers=args.decoder_num_layers,
        initial_gaussians=args.initial_gaussians,  # pass this for potential use if not resuming
        device=device,
    )

    start_iteration = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            # load model and potentially optimizer state
            model, start_iteration = GaussianChannelFieldModel.load(
                Path(args.resume),
                device,
                args,  # pass training args to load optimizer state
            )
            start_iteration += 1  # start from the next iteration
            logger.info(f"Resumed from iteration {start_iteration -1}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting from scratch.")
            args.resume = None  # ensure we don't try to resume again

    # initialize gaussians and training setup only if not resuming
    if not args.resume:
        init_pc_arg = None
        if args.init_method == "point_cloud":
            if point_cloud is not None:
                logger.info("Using point cloud for Gaussian initialization.")
                init_pc_arg = point_cloud
            else:
                logger.warning(
                    "Point cloud initialization requested but no point cloud data found. Falling back to random initialization."
                )
                args.init_method = "random"  # update arg to reflect fallback

        if args.init_method == "random":
            logger.info("Using random initialization for Gaussians.")
            # ensure env_dims is passed if available
            model.init_gaussians(
                env_dims=env_dims if env_dims is not None else None,
                point_cloud=None,  # explicitly None for random
                num_points=args.initial_gaussians,
            )
        elif args.init_method == "point_cloud":
            model.init_gaussians(
                env_dims=None,  # not needed if using point cloud
                point_cloud=init_pc_arg,
                num_points=args.initial_gaussians,  # num_points acts as max sample size here
            )

        # setup optimizer and LR schedulers
        model.training_setup(args)

    # ensure model is on the correct device
    model = model.to(device)
    logger.info(f"Model initialized/loaded with {model.get_xyz.shape[0]} Gaussians.")
    logger.info(
        f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # --- Loss Function ---
    criterion = nn.MSELoss().to(device)  # use standard MSE loss
    logger.info("Using MSE Loss for training.")

    # --- Training Loop ---
    logger.info(f"Starting training from iteration {start_iteration}...")
    progress_bar = tqdm(
        range(start_iteration, args.iterations), desc="Training GCF", dynamic_ncols=True
    )
    ema_loss = -1.0  # use -1 to indicate not yet initialized
    train_iter = iter(train_loader)  # create iterator for training data

    for iteration in progress_bar:
        iter_start_time = time.time()
        model.train()  # set model to training mode
        model.update_learning_rate(
            iteration, args
        )  # update LRs based on current iteration

        # --- Get Batch ---
        try:
            batch = next(train_iter)
        except StopIteration:
            # epoch finished, reset iterator
            train_iter = iter(train_loader)
            batch = next(train_iter)

        rx_pos_batch = batch["rx_position"].to(device)
        # target is normalized magnitude
        h_mag_gt_batch = batch["channel_magnitude"].to(device)

        # --- Add Noise to Rx Positions (Data Augmentation) ---
        if args.rx_noise_std > 0:
            noise = torch.randn_like(rx_pos_batch) * args.rx_noise_std
            rx_pos_batch = rx_pos_batch + noise

        # --- Forward Pass ---
        h_mag_pred_batch = render_magnitude(
            rx_positions=rx_pos_batch,
            model=model,
            tx_position=tx_position,
            # wavelength=wavelength, # not needed
            nt=nt,
            nr=nr,
            eps=args.snr_calc_eps,  # use snr eps for stability here too
        )

        # --- Calculate Loss ---
        mse_loss = criterion(h_mag_pred_batch, h_mag_gt_batch)
        total_loss = mse_loss
        l1_activation_loss = torch.tensor(0.0, device=device)  # initialize

        # add L1 regularization on base activation *logits* if enabled
        if args.lambda_activation_l1 > 0 and model.get_xyz.shape[0] > 0:
            # get the logits from the attribute network
            _, base_activations_logits = model.get_attributes(tx_position)
            # calculate L1 loss on the logits
            l1_activation_loss = torch.mean(torch.abs(base_activations_logits))
            total_loss = total_loss + args.lambda_activation_l1 * l1_activation_loss

        # --- Backward Pass and Optimization ---
        model.optimizer.zero_grad()  # clear previous gradients
        total_loss.backward()  # compute gradients

        # check for NaN/Inf gradients before optimizer step
        found_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and (
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
            ):
                logger.warning(
                    f"NaN or Inf gradient detected at iteration {iteration} for parameter '{name}'. Skipping optimizer step."
                )
                found_nan_grad = True
                break  # skip step if any param has bad grad

        grad_stats = {}  # initialize grad stats dict
        if not found_nan_grad:
            # compute gradient statistics (optional but useful)
            grad_stats = compute_grad_stats(model)
            # clip gradients if needed (optional, can help stability)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model.optimizer.step()  # update model parameters
        else:
            model.optimizer.zero_grad()  # clear the bad gradients if step was skipped

        # --- Logging ---
        iter_time = time.time() - iter_start_time
        with torch.no_grad():  # log metrics without tracking gradients
            current_loss = mse_loss.item()  # get scalar loss value
            # update EMA loss robustly
            if np.isnan(current_loss) or np.isinf(current_loss):
                logger.warning(
                    f"NaN or Inf loss detected at iteration {iteration}. Resetting EMA loss."
                )
                # consider stopping training or reducing LR if this happens often
                ema_loss = -1.0  # reset EMA
            elif ema_loss < 0:  # first valid loss
                ema_loss = current_loss
            else:  # update EMA
                ema_loss = 0.95 * ema_loss + 0.05 * current_loss

            # log periodically
            if iteration % args.log_freq == 0:
                # calculate SNR for logging
                snr = calculate_snr(
                    mse_loss, h_mag_gt_batch, eps=args.snr_calc_eps
                ).item()
                num_gaussians = model.get_xyz.shape[0]

                # format log message
                log_msg_train = (
                    f"[{iteration}/{args.iterations}] | "
                    f"Loss(MSE): {current_loss:.4e} | EMA Loss: {ema_loss:.4e} | "
                    f"SNR: {snr:.2f} dB | Time: {iter_time:.3f}s | Gauss: {num_gaussians}"
                )
                if args.lambda_activation_l1 > 0:
                    log_msg_train += f" | L1 Act Loss: {l1_activation_loss.item():.4e}"

                logger.info(log_msg_train)

                # log gradient stats if computed
                if grad_stats:
                    grad_log_msg = (
                        f"    Grads - Norm: {grad_stats['grad_norm']:.3e} | Mean Abs: {grad_stats['mean_abs_grad']:.3e} | "
                        f"Min: {grad_stats['min_grad']:.3e} | Max: {grad_stats['max_grad']:.3e} | "
                        f"Count: {grad_stats['param_count_with_grad']}"
                    )
                    logger.info(grad_log_msg)

                # log to tensorboard if enabled
                if writer is not None:
                    writer.add_scalar("train/mse_loss", current_loss, iteration)
                    writer.add_scalar("train/ema_loss", ema_loss, iteration)
                    writer.add_scalar("train/snr_db", snr, iteration)
                    writer.add_scalar("train/iteration_time_sec", iter_time, iteration)
                    writer.add_scalar("train/num_gaussians", num_gaussians, iteration)
                    if args.lambda_activation_l1 > 0:
                        writer.add_scalar(
                            "train/l1_activation_loss",
                            l1_activation_loss.item(),
                            iteration,
                        )
                    # log learning rates
                    if model.optimizer:
                        for i, param_group in enumerate(model.optimizer.param_groups):
                            writer.add_scalar(
                                f"lr/{param_group['name']}",
                                param_group["lr"],
                                iteration,
                            )
                    # log gradient stats
                    if grad_stats:
                        writer.add_scalar(
                            "grads/norm", grad_stats["grad_norm"], iteration
                        )
                        writer.add_scalar(
                            "grads/mean_abs", grad_stats["mean_abs_grad"], iteration
                        )
                        writer.add_scalar(
                            "grads/min", grad_stats["min_grad"], iteration
                        )
                        writer.add_scalar(
                            "grads/max", grad_stats["max_grad"], iteration
                        )

        # --- Evaluation ---
        if iteration % args.eval_freq == 0 and iteration > 0:
            if len(val_loader) > 0:  # only evaluate if val set exists
                logger.info(f"--- Starting evaluation at iteration {iteration} ---")
                eval_start_time = time.time()
                eval_metrics = evaluate(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    device=device,
                    tx_position=tx_position,
                    # wavelength=wavelength, # not needed
                    nt=nt,
                    nr=nr,
                    snr_calc_eps=args.snr_calc_eps,
                )
                eval_time = time.time() - eval_start_time
                logger.info(
                    f"Validation | Loss(MSE): {eval_metrics['val_mse_loss']:.4e} | SNR: {eval_metrics['val_snr_db']:.2f} dB | Time: {eval_time:.2f}s"
                )
                logger.info(f"--- Evaluation finished ---")

                # log validation metrics to tensorboard
                if writer is not None:
                    writer.add_scalar(
                        "validation/mse_loss", eval_metrics["val_mse_loss"], iteration
                    )
                    writer.add_scalar(
                        "validation/snr_db", eval_metrics["val_snr_db"], iteration
                    )
            else:
                logger.info(
                    f"Skipping evaluation at iteration {iteration} (validation loader is empty)."
                )

        # --- Checkpointing ---
        if (
            iteration % args.checkpoint_freq == 0 and iteration > 0
        ) or iteration == args.iterations - 1:
            checkpoint_path = checkpoints_dir / f"checkpoint_{iteration:07d}.pt"
            model.save(checkpoint_path, iteration=iteration)
            logger.info(f"Checkpoint saved to {checkpoint_path}")

    # --- Training Finished ---
    progress_bar.close()  # close the tqdm bar
    final_model_path = log_dir / "final_model.pt"
    model.save(final_model_path, iteration=args.iterations - 1)
    logger.info(f"Training completed after {args.iterations} iterations.")
    logger.info(f"Final model saved to {final_model_path}")

    if writer is not None:
        writer.close()  # close tensorboard writer


if __name__ == "__main__":
    args = parse_args()
    train(args)
```
