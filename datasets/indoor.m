%% environment setup
close all force; clear; clc;

mapFileName = "models/conference.stl";
[stl_data, ~] = stlread(mapFileName);
viewer = siteviewer("SceneModel", mapFileName, "Transparency", 0.25);

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

%% extra config
plot_rays = false;
visualize = true;

% generate point cloud
point_cloud = generate_pc(vertices, faces, pc_params, env_dims, visualize);

%% system config
fc = 5e9;
lambda = physconst("lightspeed") / fc;
num_tx_ant = 16;
num_rx_ant = 2;

% OFDM parameters
cfg = wlanNonHTConfig;
cfg.ChannelBandwidth = 'CBW80';

txArray = arrayConfig("Size", [num_tx_ant / 4 num_tx_ant / 4], "ElementSpacing", lambda / 2);
rxArray = arrayConfig("Size", [1 num_rx_ant], "ElementSpacing", lambda / 2);

%% AP setup
AP = txsite("cartesian", ...
    "Antenna", txArray, ...
    "AntennaPosition", [-1.5; 0.0; 2.1], ... % Positioned near ceiling
    "TransmitterFrequency", fc, ...
    "TransmitterPower", 0.05);

%% user setup
approx_target_users = 418;

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

if method == "image"
    max_refs = 2;
else
    max_refs = 3;
end

pm = propagationModel("raytracing", ...
    "Method", method, ...
    "CoordinateSystem", "cartesian", ...
    "SurfaceMaterial", "wood", ...
    "TerrainMaterial", "wood", ...
    "MaxNumReflections", max_refs, ...
    "UseGPU", "on");

rays = raytrace(AP, Users, pm, "Map", mapFileName);

% Filter users to keep only those with valid rays
valid_user_mask = ~cellfun(@isempty, rays);
Users = Users(valid_user_mask);
rays = rays(valid_user_mask);
num_users = sum(valid_user_mask);

if num_users < approx_target_users
    warning('Only %d out of %d users had valid rays, discarding the rest.', ...
        num_users, approx_target_users);
end

%% visualize
show(AP, "ShowAntennaHeight", false)
show(Users, "ShowAntennaHeight", false)

if plot_rays
    for userIdx = 1:(num_users / 20) % ~20 % rays
        plot(rays{userIdx}, "Colormap", jet)
        pause(0.05)
    end
end

%% extract positions
rx_positions = zeros(3, num_users);

for userIdx = 1:num_users
    rx_positions(:, userIdx) = Users(userIdx).AntennaPosition;
end

%% CSI collection
ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
numSubcarriers = length(ofdmInfo.ActiveFrequencyIndices);

sc_spacing = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;
freqs = fc + ofdmInfo.ActiveFrequencyIndices * sc_spacing;

H = zeros(num_users, num_tx_ant, num_rx_ant, numSubcarriers);
AoD_all = cell(num_users, 1);
AoA_all = cell(num_users, 1);

for userIdx = 1:num_users
    [H(userIdx, :, :, :), AoD_all{userIdx}, AoA_all{userIdx}] = ...
        generate_csi(rays{userIdx}, fc, cfg, num_tx_ant, num_rx_ant, method, 'indoor');
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

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

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

filename = sprintf('%s/conf_%dx%d_%du_%.1fghz_%sRT.mat', ...
    output_dir, ...
    num_tx_ant, ...
    num_rx_ant, ...
    num_users, ...
    fc / 1e9, ...
    method);

save(filename, 'dataset', '-v7.3');
