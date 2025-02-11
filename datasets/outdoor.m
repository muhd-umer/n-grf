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
num_tx_ant = 8 * 8;
num_rx_ant = 2;

% OFDM parameters
carrier = nrCarrierConfig;
carrier.SubcarrierSpacing = 15;
carrier.NSizeGrid = 52;
cfg = carrier;

% extra config
use_single_sc = true;
sc_idx = [];

txArray = arrayConfig("Size", [8 8], "ElementSpacing", lambda);
rxArray = arrayConfig("Size", [1 num_rx_ant], "ElementSpacing", lambda / 2);

%% AP setup
AP = txsite("cartesian", ...
    "Antenna", txArray, ...
    "AntennaPosition", [20; 38; 20], ...
    "TransmitterFrequency", fc, ...
    "TransmitterPower", 10);

%% user setup
approx_target_users = 3147;

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

for userIdx = 1:num_users
    [H(userIdx, :, :, :), AoD_all{userIdx}, AoA_all{userIdx}] = ...
        generate_csi(rays{userIdx}, fc, cfg, num_tx_ant, num_rx_ant, method, 'outdoor', use_single_sc, sc_idx);
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

dataset.environment.dimensions = env_dims;
dataset.environment.point_cloud = point_cloud;
dataset.environment.pc_params = pc_params;

dataset.nodes.ap_position = AP.AntennaPosition';
dataset.nodes.users_positions = rx_positions;

dataset.channel.H = H;
dataset.channel.AoD = AoD_all;
dataset.channel.AoA = AoA_all;
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
