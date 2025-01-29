%% Environment Setup
close all force; clear; clc;

mapFileName = "models/conference.stl";
viewer = siteviewer("SceneModel", mapFileName, "Transparency", 0.25);

[vertex, face] = stlread(mapFileName);
xy_offset = 0.1;
z_offset = 0.1;
env_dims = [
            [min(vertex.Points(:, 1)) + xy_offset, max(vertex.Points(:, 1)) - xy_offset];
            [min(vertex.Points(:, 2)) + xy_offset, max(vertex.Points(:, 2)) - xy_offset];
            [min(vertex.Points(:, 3)), max(vertex.Points(:, 3)) - z_offset]
            ];

%% System config
fc = 5.8e9;
lambda = physconst("lightspeed") / fc;
num_tx_ant = 16;
num_rx_ant = 2;

% OFDM parameters
cfg = wlanNonHTConfig;
cfg.ChannelBandwidth = 'CBW20'; % 20 MHz bandwidth

txArray = arrayConfig("Size", [num_tx_ant / 4 num_tx_ant / 4], "ElementSpacing", lambda / 2);
rxArray = arrayConfig("Size", [1 num_rx_ant], "ElementSpacing", lambda / 2);

%% AP setup
AP = txsite("cartesian", ...
    "Antenna", txArray, ...
    "AntennaPosition", [-1.5; 0.0; 2.1], ... % Positioned near ceiling
    "TransmitterFrequency", fc, ...
    "TransmitterPower", 0.05); % 50mW transmit power

%% User setup
distribution = "random"; % "uniform" | "random"
numUsers = 1000; % Number of users to simulate
userSeparation = 0.5; % Minimum separation in meters (for uniform)

% seed
S = RandStream("mt19937ar", "Seed", 5489);
RandStream.setGlobalStream(S);

if distribution == "uniform"
    Users = create_users(env_dims, userSeparation, rxArray, "uniform");
else
    Users = create_users(env_dims, numUsers, rxArray, "random");
end

%% visualize
show(AP, "ShowAntennaHeight", false)
show(Users, "ShowAntennaHeight", false)

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
    "MaxNumReflections", max_refs, ...
    "UseGPU", "on");

rays = raytrace(AP, Users, pm, "Map", mapFileName);

%% plot rays
for userIdx = 1:(numUsers / 10) % plot only 10 % of the users

    if ~isempty(rays{userIdx})
        plot(rays{userIdx}, "Colormap", jet, "ColorLimits", [50, 95])
        pause(0.05)
    end

end

%% extract positions
rx_positions = zeros(3, numUsers);

for userIdx = 1:numUsers
    rx_positions(:, userIdx) = Users(userIdx).AntennaPosition;
end

%% CSI collection
ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
numSubcarriers = length(ofdmInfo.ActiveFrequencyIndices);

sc_spacing = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;
freqs = fc + ofdmInfo.ActiveFrequencyIndices * sc_spacing;

H = zeros(numUsers, num_tx_ant, num_rx_ant, numSubcarriers);
AoD_all = cell(numUsers, 1);
AoA_all = cell(numUsers, 1);

for userIdx = 1:numUsers
    if ~isempty(rays{userIdx})
        [H(userIdx, :, :, :), AoD_all{userIdx}, AoA_all{userIdx}] = ...
            generate_csi(rays{userIdx}, fc, cfg, num_tx_ant, num_rx_ant, method);
    end

    if mod(userIdx, floor(numUsers / 5)) == 0
        disp(['Running ... ', num2str(round(100 * userIdx / numUsers)), '%']);
    end
end

%% ray marching and per-ray information (for NeWRF comparison)
ray_steps = cell(numUsers, 1);
ray_points = cell(numUsers, 1);
ray_interactions = cell(numUsers, 1);
ray_coefficients = cell(numUsers, 1);

for userIdx = 1:numUsers

    if ~isempty(rays{userIdx})
        [ray_steps{userIdx}, ray_points{userIdx}] = ray_marching(rays{userIdx});
        [ray_interactions{userIdx}, ray_coefficients{userIdx}] = get_ray_chan(rays{userIdx}, freqs, method);
    end

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
dataset.config.distribution = distribution;
dataset.config.num_users = numUsers;
dataset.config.method = method;
dataset.config.ofdm = cfg;

dataset.environment.map_file = mapFileName;
dataset.environment.dimensions = env_dims;

dataset.nodes.ap = struct('position', AP.AntennaPosition', ...
    'array', txArray);
dataset.nodes.users = struct('positions', rx_positions, ...
    'array', rxArray);

dataset.channel.H = H;
dataset.channel.AoD = AoD_all;
dataset.channel.AoA = AoA_all;
dataset.channel.ray_steps = ray_steps;
dataset.channel.ray_points = ray_points;
dataset.channel.ray_interactions = ray_interactions;
dataset.channel.ray_coefficients = ray_coefficients;
dataset.channel.frequencies = freqs;

filename = sprintf('%s/%s_%dx%d_%s%du_%.1fghz_%sRT.mat', ...
    output_dir, ...
    mapname, ...
    num_tx_ant, ...
    num_rx_ant, ...
    distribution, ...
    numUsers, ...
    fc / 1e9, ...
    method);

save(filename, 'dataset');
