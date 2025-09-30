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
approx_target_users = 1316;

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

%% trajectory generation for non-stationary scenario
% trajectory parameters
time_step = 0.5; % time step in seconds for position updates
max_updates = 10; % maximum number of position updates
min_speed = 0.5; % minimum speed in m/s
max_speed = 2.0; % maximum speed in m/s

% generate random trajectories for each user
num_users = actual_users;
trajectories = struct('direction', cell(num_users, 1), 'speed', cell(num_users, 1));

for userIdx = 1:num_users
    % random direction in xy plane (angle in radians)
    angle = unifrnd(0, 2 * pi);
    trajectories(userIdx).direction = [cos(angle); sin(angle); 0];

    % random speed
    trajectories(userIdx).speed = unifrnd(min_speed, max_speed);
end

%% output directory setup
output_base_dir = "outputs/outdoor_ns";
mkdir(output_base_dir);

% get base filename components
[~, mapname] = fileparts(mapFileName);
sc_str = '';

if use_single_sc
    sc_str = sprintf('_sc%d', sc_idx);
end

base_filename = sprintf('iab_%dx%d_%du_%.1fghz_%sRT%s', ...
    num_tx_ant, ...
    num_rx_ant, ...
    num_users, ...
    fc / 1e9, ...
    "sbr", ...
    sc_str);

% create folder for this dataset
dataset_folder = fullfile(output_base_dir, base_filename);
mkdir(dataset_folder);

%% RT simulation method setup
method = "sbr"; % "image" | "sbr"
max_refs = 1;

pm = propagationModel("raytracing", ...
    "Method", method, ...
    "CoordinateSystem", "cartesian", ...
    "MaxNumDiffractions", 1, ...
    "MaxNumReflections", max_refs, ...
    "UseGPU", "auto");

%% CSI collection setup
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

%% monte carlo loop: generate channels at different positions
fprintf('Starting simulation for non-stationary channels...\n');

% keep track of valid users across updates
valid_users_mask = true(num_users, 1);
Users_current = Users;

for update_idx = 0:max_updates
    fprintf('\nUpdate %d/%d\n', update_idx, max_updates);

    % count valid users for this update
    num_valid_users = sum(valid_users_mask);

    if num_valid_users == 0
        warning('No valid users remaining. Stopping Monte Carlo loop...');
        break;
    end

    fprintf('Valid users: %d/%d\n', num_valid_users, num_users);

    %% run raytracing for current positions
    rays = raytrace(AP, Users_current, pm, "Map", mapFileName, "Type", "pathloss");

    % filter to keep only users with valid rays
    current_valid_mask = ~cellfun(@isempty, rays);
    combined_valid_mask = valid_users_mask & current_valid_mask;

    Users_valid = Users_current(combined_valid_mask);
    rays_valid = rays(combined_valid_mask);
    num_valid_rays = sum(combined_valid_mask);

    fprintf('Users with valid rays: %d/%d\n', num_valid_rays, num_valid_users);

    if num_valid_rays == 0
        warning('No users have valid rays at update %d. Stopping Monte Carlo loop.', update_idx);
        break;
    end

    %% visualize (optional)
    if visualize && update_idx == 0
        show(AP, "ShowAntennaHeight", false)
        show(Users_valid, "ShowAntennaHeight", false)

        if plot_rays

            for userIdx = 1:(num_valid_rays / 20) % ~20 rays
                plot(rays_valid{userIdx}, "Colormap", jet)
                pause(0.05)
            end

        end

    end

    %% extract positions
    rx_positions = zeros(3, num_valid_rays);

    for userIdx = 1:num_valid_rays
        rx_positions(:, userIdx) = Users_valid(userIdx).AntennaPosition;
    end

    %% generate CSI
    H = zeros(num_valid_rays, num_tx_ant, num_rx_ant, numSubcarriers);
    AoD_all = cell(num_valid_rays, 1);
    AoA_all = cell(num_valid_rays, 1);
    path_loss = zeros(num_valid_rays, 1);
    path_loss_per_ray = cell(num_valid_rays, 1);

    for userIdx = 1:num_valid_rays
        [H(userIdx, :, :, :), AoD_all{userIdx}, AoA_all{userIdx}] = ...
            generate_csi(rays_valid{userIdx}, fc, cfg, txArray, rxArray, method, 'outdoor', use_single_sc, sc_idx);
        path_loss(userIdx) = mean([rays_valid{userIdx}.PathLoss]);
        path_loss_per_ray{userIdx} = [rays_valid{userIdx}.PathLoss];
    end

    % check for null values in channel matrix
    if any(isnan(H(:)))
        warning('Channel matrix contains NaN values at update %d!', update_idx);
    end

    %% ray marching and per-ray information
    ray_steps = cell(num_valid_rays, 1);
    ray_points = cell(num_valid_rays, 1);
    ray_interactions = cell(num_valid_rays, 1);
    ray_coefficients = cell(num_valid_rays, 1);

    for userIdx = 1:num_valid_rays
        [ray_steps{userIdx}, ray_points{userIdx}] = ray_marching(rays_valid{userIdx});
        [ray_interactions{userIdx}, ray_coefficients{userIdx}] = get_ray_chan(rays_valid{userIdx}, freqs, method);
    end

    % check for null values in ray marching results
    if any(cellfun(@(x) any(cellfun(@(y) any(isnan(y(:))), x)), ray_steps)) || ...
            any(cellfun(@(x) any(cellfun(@(y) any(isnan(y(:))), x)), ray_points)) || ...
            any(cellfun(@(x) any(isnan(x(:))), ray_coefficients))
        warning('Ray marching results contain NaN values at update %d!', update_idx);
    end

    %% create dataset structure
    dataset = struct();

    dataset.config.tx_antennas = num_tx_ant;
    dataset.config.rx_antennas = num_rx_ant;
    dataset.config.frequency = fc;
    dataset.config.wavelength = lambda;
    dataset.config.num_users = num_valid_rays;
    dataset.config.use_siso = use_siso;

    dataset.environment.dimensions = env_dims;
    dataset.environment.point_cloud = point_cloud;
    dataset.environment.pc_params = pc_params;

    dataset.nodes.ap_position = AP.AntennaPosition';
    dataset.nodes.users_positions = rx_positions;

    dataset.channel.H = H;
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

    % add non-stationary specific info
    dataset.trajectory.update_number = update_idx;
    dataset.trajectory.time_step = time_step;
    dataset.trajectory.elapsed_time = update_idx * time_step;

    %% save dataset
    if update_idx == 0
        % base dataset (initial positions)
        filename = sprintf('%s/%s_base.mat', dataset_folder, base_filename);
    else
        % update datasets
        filename = sprintf('%s/%s_u%d.mat', dataset_folder, base_filename, update_idx);
    end

    save(filename, 'dataset', '-v7.3');
    fprintf('Saved dataset to: %s\n', filename);

    %% update positions along trajectories
    if update_idx < max_updates
        fprintf('Updating user positions...\n');

        % create array for updated users
        Users_next = Users_current;

        for userIdx = 1:num_users

            if ~valid_users_mask(userIdx)
                continue;
            end

            % get current position
            current_pos = Users_current(userIdx).AntennaPosition;

            % calculate displacement
            displacement = trajectories(userIdx).direction * trajectories(userIdx).speed * time_step;

            % new position
            new_pos = current_pos + displacement;

            % check if new position is within environment bounds (xy only, keep z the same)
            if new_pos(1) < env_dims(1, 1) || new_pos(1) > env_dims(1, 2) || ...
                    new_pos(2) < env_dims(2, 1) || new_pos(2) > env_dims(2, 2)
                % user has left the environment, mark as invalid
                valid_users_mask(userIdx) = false;
                fprintf('User %d left environment bounds.\n', userIdx);
            else
                % update user position (keep z coordinate the same for ground-level users)
                new_pos(3) = current_pos(3);
                Users_next(userIdx).AntennaPosition = new_pos;
            end

        end

        Users_current = Users_next;
        fprintf('Position update complete. Remaining valid users: %d/%d\n', sum(valid_users_mask), num_users);
    end

end

fprintf('\nMonte Carlo simulation complete!\n');
fprintf('Datasets saved to: %s\n', dataset_folder);

%% helper function for truncating data
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
