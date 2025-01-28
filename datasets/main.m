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
    "TransmitterPower", 0.1); % 100mW transmit power

%% User setup
distribution = "random"; % "uniform" | "random"
numUsers = 50; % Number of users to simulate
userSeparation = 0.5; % Minimum separation in meters (for uniform)

% seed
S = RandStream("mt19937ar", "Seed", 5489);
RandStream.setGlobalStream(S);

if distribution == "uniform"
    Users = createUsers(env_dims, userSeparation, rxArray, "uniform");
else
    Users = createUsers(env_dims, numUsers, rxArray, "random");
end

%% Visualize
show(AP, "ShowAntennaHeight", false)
show(Users, "ShowAntennaHeight", false)

%% RT Simulation
pm = propagationModel("raytracing", ...
    "Method", "image", ...
    "CoordinateSystem", "cartesian", ...
    "SurfaceMaterial", "wood", ...
    "MaxNumReflections", 2);

rays = raytrace(AP, Users, pm, "Map", mapFileName);

%% Plot rays
siteviewer("SceneModel", mapFileName);
show(AP, "ShowAntennaHeight", false)
show(Users, "ShowAntennaHeight", false)

% plot ray of a few users
for userIdx = 1:(numUsers/10)
    if ~isempty(rays{userIdx})
        plot(rays{userIdx}, "Colormap", jet, "ColorLimits", [50, 95])
        pause(0.05)
    end
end

%% CSI collection
ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
numSubcarriers = length(ofdmInfo.ActiveFrequencyIndices);
numUsers = length(Users);

H = zeros(numUsers, num_tx_ant, num_rx_ant, numSubcarriers);

for userIdx = 1:numUsers

    if ~isempty(rays{userIdx})
        H(userIdx, :, :, :) = generateCSI(rays{userIdx}, fc, cfg, num_tx_ant, num_rx_ant);
    end

    if mod(userIdx, floor(numUsers / 5)) == 0
        disp(['Running ... ', num2str(round(100 * userIdx / numUsers)), '%']);
    end

end

%% save
output_dir = "outputs";

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Extract map name from path without extension
[~, mapname] = fileparts(mapFileName);

filename = sprintf('%s/%s_%dx%d_%s%du_%.1fghz.mat', ...
    output_dir, ...
    mapname, ...
    num_tx_ant, ...
    num_rx_ant, ...
    distribution, ...
    numUsers, ...
    fc / 1e9);

save(filename, 'H', 'AP', 'Users', 'cfg', 'num_tx_ant', 'num_rx_ant');
