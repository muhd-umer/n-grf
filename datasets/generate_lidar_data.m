%% Lidar Point Cloud Generation
function generate_lidar_data(mapFileName, output_dir)
% GENERATE_LIDAR_DATA Creates 3D point cloud from environment model
%   mapFileName: Path to STL file (same as wireless simulation)
%   output_dir: Directory to save point cloud data

%% Setup environment (consistent with wireless simulation)
close all force; clc;

% Read environment dimensions from STL (same as indoor.m)
[vertex, face] = stlread(mapFileName);
xy_offset = 0.1;
z_offset = 0.1;
env_dims = [
    [min(vertex.Points(:,1)) + xy_offset, max(vertex.Points(:,1)) - xy_offset];
    [min(vertex.Points(:,2)) + xy_offset, max(vertex.Points(:,2)) - xy_offset];
    [min(vertex.Points(:,3)), max(vertex.Points(:,3)) - z_offset]
];

%% LiDAR parameters
horizontal_res = 0.1;        % Horizontal resolution in degrees
vertical_res = max(1, round(0.1));  % Ensures it's at least 1
vertical_foV = 60;           % Total vertical field of view (degrees)
max_range = 15;              % Maximum detection range (meters)
sensor_position = [-1.5, 0.0, 2.1]; % Same as AP position

%% Create virtual LiDAR sensor using valid syntax
lidar = lidarParameters(vertical_res, vertical_foV, horizontal_res);
lidar.RangeAccuracy = 0.01;  % Set additional parameters
lidar.MaxRange = max_range;

%% Generate spherical scanning pattern (aligned with parameters)
azimuth = -180:horizontal_res:180;
elevation = -vertical_foV/2:vertical_res:vertical_foV/2;


% Initialize point cloud
pointCloud = [];

%% Ray casting simulation
for az = azimuth
    for el = elevation
        % Calculate ray direction
        [x,y,z] = sph2cart(deg2rad(az), deg2rad(el), max_range);
        ray_end = sensor_position + [x,y,z];
        
        % Check intersection with environment
        [intersectPt, ~] = rayIntersection(...
            scene,...
            [sensor_position; ray_end],...
            'World');
        
        % Add valid points to cloud
        if ~isnan(intersectPt)
            % Verify point is within environment bounds
            if all(intersectPt >= env_dims(:,1)') && ...
               all(intersectPt <= env_dims(:,2)')
                pointCloud = [pointCloud; intersectPt];
            end
        end
    end
end

%% Post-processing
% Remove duplicate points
pointCloud = unique(pointCloud, 'rows');

% Add intensity simulation (distance-based)
intensity = 1 - vecnorm(pointCloud - sensor_position, 2, 2)/max_range;
lidarData = pointCloud(:,1:3);
lidarData(:,4) = intensity;

%% Save data in same format as wireless dataset
[~, mapname] = fileparts(mapFileName);
output_file = fullfile(output_dir, sprintf('%s_lidar.mat', mapname));

% Create structure matching wireless data format
lidarDataset = struct();
lidarDataset.map_file = mapFileName;
lidarDataset.sensor_position = sensor_position;
lidarDataset.point_cloud = lidarData;
lidarDataset.parameters = lidar;

save(output_file, 'lidarDataset');
fprintf('LiDAR data saved to: %s\n', output_file);

%% Visualize
figure;
pcshow(lidarData(:,1:3), lidarData(:,4));
title('Simulated LiDAR Point Cloud');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
colormap(jet); colorbar; axis equal;

end