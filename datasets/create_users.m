function [Users, num_created] = create_users(env_dims, num_users, rx_array, user_params, stl_data)
    %CREATE_USERS Generate user positions in a 3D environment
    %   [USERS, NUM_CREATED] = CREATE_USERS(ENV_DIMS, NUM_USERS, RX_ARRAY, USER_PARAMS, STL_DATA)
    %   creates user receiver sites within specified environmental dimensions.
    %
    %   Inputs:
    %       ENV_DIMS - 2x2 matrix defining environment bounds [xmin xmax; ymin ymax]
    %       NUM_USERS - Number of users to generate
    %       RX_ARRAY - Antenna array configuration for receivers
    %       USER_PARAMS - Structure with optional parameters:
    %           .check_building_collision (logical) - Enable building collision detection
    %           .check_user_collision (logical) - Enable user separation checks
    %           .separation_distance (double) - Minimum distance between users
    %       STL_DATA - STL mesh data for building collision detection (optional)
    %
    %   Outputs:
    %       USERS - Array of rxsite objects representing user positions
    %       NUM_CREATED - Actual number of users created (may be less than NUM_USERS)
    %
    %   Example:
    %       env_dims = [-10 10; -10 10];
    %       num_users = 10;
    %       rx_array = phased.ULA;
    %       user_params.check_building_collision = true;
    %       user_params.check_user_collision = true;
    %       user_params.separation_distance = 2;
    %       [users, created] = create_users(env_dims, num_users, rx_array, user_params, []);

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
    max_attempts = num_users * 100; % arbitrary limit to avoid infinite loop
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
