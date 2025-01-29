function [step_lengths, ray_points] = ray_marching(rays)
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
        step_lengths{i} = sqrt(sum(point_diffs.^2));
    end
end
