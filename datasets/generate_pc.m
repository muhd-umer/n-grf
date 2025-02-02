function all_points = generate_pc(vertices, faces, params, env_dims, visualize)
    %GENERATE_PC Generate point clouds for a given environment
    %   POINTS = GENERATE_PC(VERTICES, FACES, PARAMS, ENV_DIMS, VISUALIZE)
    %   generates a point cloud for a GIVEN environment using various
    %   sampling strategies.
    %
    %   Inputs:
    %       VERTICES    - Nx3 matrix of vertex coordinates from STL
    %       FACES      - Mx3 matrix of face indices from STL
    %       PARAMS     - Structure containing point generation parameters:
    %           .edge_density      - Points per edge unit length (default: 0)
    %           .surface_density   - Points per triangle unit area (default: 0)
    %           .volume_density    - Points per unit volume (default: 0)
    %           .boundary_density  - Points per boundary surface area (default: 0)
    %           .random_points     - Additional random points in volume (default: 0)
    %           .noise_std        - Standard deviation for point perturbation (default: 0)
    %           .edge_reduction    - Factor to reduce edge points (default: 1)
    %           .surface_reduction - Factor to reduce surface points (default: 1)
    %       ENV_DIMS   - 3x2 matrix of environment dimensions [min_x max_x;
    %                                                        min_y max_y;
    %                                                        min_z max_z]
    %       VISUALIZE  - (Optional) Boolean to display point cloud, default false
    %
    %   Output:
    %       ALL_POINTS - Px3 matrix of generated points combining edge, surface,
    %                   boundary, and volume points
    %
    %   See also STLREAD, POINTCLOUD, PCSHOW

    % set default values if not provided
    default_params = struct('edge_density', 0, ...
        'surface_density', 0, ...
        'volume_density', 0, ...
        'boundary_density', 0, ...
        'random_points', 0, ...
        'noise_std', 0, ...
        'edge_reduction', 1, ...
        'surface_reduction', 1);

    % merge provided params with defaults
    if ~isfield(params, 'edge_density'), params.edge_density = default_params.edge_density; end
    if ~isfield(params, 'surface_density'), params.surface_density = default_params.surface_density; end
    if ~isfield(params, 'volume_density'), params.volume_density = default_params.volume_density; end
    if ~isfield(params, 'boundary_density'), params.boundary_density = default_params.boundary_density; end
    if ~isfield(params, 'random_points'), params.random_points = default_params.random_points; end
    if ~isfield(params, 'noise_std'), params.noise_std = default_params.noise_std; end
    if ~isfield(params, 'edge_reduction'), params.edge_reduction = default_params.edge_reduction; end
    if ~isfield(params, 'surface_reduction'), params.surface_reduction = default_params.surface_reduction; end

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

    if visualize
        pt_cloud = pointCloud(all_points);
        figure;
        pcshow(pt_cloud);
        title(sprintf('total points: %d', size(all_points, 1)));
        xlabel('x'); ylabel('y'); zlabel('z');
        grid on;
    end

end
