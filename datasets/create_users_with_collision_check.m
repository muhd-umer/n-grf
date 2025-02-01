function Users = create_users_with_collision_check(env_dims, numUsers, rxArray, distribution, stl_data, tx_height)
    % Initialize output array
    Users(numUsers) = rxsite;
    
    % Get vertices and faces from STL
    vertices = stl_data.Points;
    faces = stl_data.ConnectivityList;
    
    % Counter for valid users
    valid_users = 0;
    max_attempts = numUsers * 100;  % Prevent infinite loops
    attempts = 0;
    
    % Define height parameters
    min_height = env_dims(3,1);  % Minimum height from environment
    max_height = min(tx_height, env_dims(3,2));  % Don't exceed transmitter height
    
    % Define a probability distribution that favors lower heights
    % This creates an exponential decay probability for height selection
    height_decay_rate = 0.5;  % Adjust this to control how quickly probability decreases with height
    
    while valid_users < numUsers && attempts < max_attempts
        % Generate random position within environment dimensions
        x = env_dims(1,1) + (env_dims(1,2) - env_dims(1,1)) * rand();
        y = env_dims(2,1) + (env_dims(2,2) - env_dims(2,1)) * rand();
        
        % Generate height using exponential distribution
        height_range = max_height - min_height;
        random_exp = -log(1 - rand()) / height_decay_rate;
        normalized_height = min(random_exp, 1);  % Clip to 1
        z = min_height + normalized_height * height_range;
        
        % Add some probability of being exactly at ground level
        if rand() < 0.3  % 30% chance of being at ground level
            z = min_height;
        end
        
        point = [x, y, z];
        
        if ~is_point_in_building(point, vertices, faces)
            valid_users = valid_users + 1;
            Users(valid_users) = rxsite("cartesian", ...
                "Antenna", rxArray, ...
                "AntennaPosition", point');
        end
        
        attempts = attempts + 1;
    end
    
    if valid_users < numUsers
        warning('Could not place all users outside buildings after %d attempts', max_attempts);
        Users = Users(1:valid_users);
    end
end

function inside = is_point_in_building(point, vertices, faces)
    % Ray casting algorithm for point-in-mesh detection
    % Cast a ray in positive x direction and count intersections
    
    ray_direction = [1, 0, 0];
    intersections = 0;
    
    for i = 1:size(faces, 1)
        triangle = vertices(faces(i, :), :);
        if does_ray_intersect_triangle(point, ray_direction, triangle)
            intersections = intersections + 1;
        end
    end
    
    % If number of intersections is odd, point is inside
    inside = mod(intersections, 2) == 1;
end

function intersects = does_ray_intersect_triangle(origin, direction, triangle)
    % Möller–Trumbore ray-triangle intersection algorithm
    epsilon = 1e-7;
    
    edge1 = triangle(2,:) - triangle(1,:);
    edge2 = triangle(3,:) - triangle(1,:);
    h = cross(direction, edge2);
    a = dot(edge1, h);
    
    if abs(a) < epsilon
        intersects = false;
        return;
    end
    
    f = 1.0/a;
    s = origin - triangle(1,:);
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