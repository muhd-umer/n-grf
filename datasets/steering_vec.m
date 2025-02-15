function [v_tx, v_rx] = steering_vec(angles, txArray, rxArray, lambda)
    % STEERING_VEC Compute array steering vectors for MIMO systems
    %
    % Description:
    %   Computes transmit and receive array steering vectors for a MIMO system
    %   with a Uniform Rectangular Array (URA) at transmitter and Uniform Linear
    %   Array (ULA) at receiver.
    %
    % Inputs:
    %   angles   - [2×1] vector containing [azimuth; elevation] in degrees
    %   txArray  - phased.URA object for transmit array configuration
    %   rxArray  - phased.ULA object for receive array configuration
    %   lambda   - Scalar wavelength in meters
    %
    % Outputs:
    %   v_tx     - [(M×N)×1] transmit steering vector for URA of size M×N
    %   v_rx     - [P×1] receive steering vector for ULA of size P
    %
    % Example:
    %   angles = [30; 45];  % azimuth = 30°, elevation = 45°
    %   txArray = phased.URA('Size', [4 4], 'ElementSpacing', [0.5 0.5]);
    %   rxArray = phased.ULA('NumElements', 8, 'ElementSpacing', 0.5);
    %   lambda = 0.1;  % 10cm wavelength
    %   [v_tx, v_rx] = steering_vec(angles, txArray, rxArray, lambda);

    angles_rad = deg2rad(angles);

    if isa(txArray, 'phased.URA')
        v_tx = steering_vector_tx(angles_rad, txArray.Size, txArray.ElementSpacing, lambda);
    else
        error('Transmit array must be a phased.URA object');
    end

    if isa(rxArray, 'phased.ULA')
        v_rx = steering_vector_rx(angles_rad, rxArray.NumElements, rxArray.ElementSpacing, lambda);
    else
        error('Receive array must be a phased.ULA object');
    end

end

function v = steering_vector_tx(angles, array_size, element_spacing, lambda)
    % STEERING_VECTOR_TX Compute steering vector for Uniform Rectangular Array
    %
    % Description:
    %   Computes the steering vector for a Uniform Rectangular Array (URA) given
    %   the angles of arrival/departure, array size, element spacing, and wavelength.
    %
    % Inputs:
    %   angles          - [2×1] vector [azimuth; elevation] in radians
    %   array_size      - [1×2] vector [M N] specifying URA dimensions
    %   element_spacing - [1×2] vector specifying spacing between array elements [dx dy]
    %   lambda          - Scalar wavelength in meters
    %
    % Output:
    %   v               - [(M×N)×1] steering vector for URA
    %
    % Example:
    %   angles = [pi/6; pi/4];  % azimuth = 30°, elevation = 45°
    %   array_size = [4 4];
    %   element_spacing = [0.5 0.5];
    %   lambda = 0.1;  % 10cm wavelength
    %   v = steering_vector_tx(angles, array_size, element_spacing, lambda);

    az = angles(1);
    el = angles(2);

    M = array_size(1);
    N = array_size(2);

    [x_grid, y_grid] = meshgrid(0:M - 1, 0:N - 1);
    x_positions = (x_grid(:) - (M - 1) / 2) * element_spacing(1);
    y_positions = (y_grid(:) - (N - 1) / 2) * element_spacing(2);

    % compute wavenumber (k = 2π/λ)
    k = 2 * pi / lambda;
    v = exp(-1j * k * (x_positions * cos(el) * cos(az) + ...
        y_positions * cos(el) * sin(az)));
end

function v = steering_vector_rx(angles, num_elements, element_spacing, lambda)
    % STEERING_VECTOR_RX Compute steering vector for Uniform Linear Array
    %
    % Description:
    %   Computes the steering vector for a Uniform Linear Array (ULA) given
    %   the angles of arrival/departure, number of elements, element spacing, and wavelength.
    %
    % Inputs:
    %   angles          - [2×1] vector [azimuth; elevation] in radians
    %   num_elements    - Scalar number of elements in the ULA
    %   element_spacing - Scalar spacing between array elements
    %   lambda          - Scalar wavelength in meters
    %
    % Output:
    %   v               - [P×1] steering vector for ULA, where P is num_elements
    %
    % Example:
    %   angles = [pi/6; pi/4];  % azimuth = 30°, elevation = 45°
    %   num_elements = 8;
    %   element_spacing = 0.5;
    %   lambda = 0.1;  % 10cm wavelength
    %   v = steering_vector_rx(angles, num_elements, element_spacing, lambda);

    az = angles(1);
    el = angles(2);

    indices = (0:num_elements - 1).';
    x_positions = (indices - (num_elements - 1) / 2) * element_spacing;

    % compute wavenumber (k = 2π/λ)
    k = 2 * pi / lambda;
    v = exp(-1j * k * (x_positions * cos(el) * cos(az)));
end
