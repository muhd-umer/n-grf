function [interaction_points, ray_coeffs] = get_ray_chan(rays, freqs, method)
    n_paths = length(rays);
    n_freqs = length(freqs);

    interaction_points = zeros(3, n_paths);
    ray_coeffs = zeros(n_freqs, n_paths);

    for i = 1:n_paths

        if rays(i).LineOfSight
            points = [rays(i).TransmitterLocation rays(i).ReceiverLocation];
        else
            points = [rays(i).TransmitterLocation rays(i).Interactions.Location rays(i).ReceiverLocation];
        end

        interaction_points(:, i) = points(:, end - 1);

        for j = 1:n_freqs
            f = freqs(j);

            if strcmp(method, "sbr")
                path_loss = rays(i).PathLoss;
                phase = rays(i).PhaseShift;
            else
                prop_dist = rays(i).PropagationDistance;
                path_loss = fspl(prop_dist, f);
                phase = 2 * pi * f * prop_dist / physconst("lightspeed");
            end

            ray_coeffs(j, i) = 10 ^ (-path_loss / 20) * exp(-1j * phase);
        end

    end

end
