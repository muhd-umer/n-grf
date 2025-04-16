function [v_tx, v_rx] = steering_vec(aod_angles, aoa_angles, txArray, rxArray, lambda)
    % STEERING_VEC Compute array steering vectors for MIMO systems
    %
    % Description:
    %   Computes transmit and receive array steering vectors for a MIMO system
    %   with a Uniform Rectangular Array (URA) at transmitter and Uniform Linear
    %   Array (ULA) at receiver. Uses Angle of Departure (AoD) for Tx array
    %   and Angle of Arrival (AoA) for Rx array.
    %
    % Inputs:
    %   aod_angles - [2x1] vector containing Tx [azimuth; elevation] in degrees (AoD)
    %   aoa_angles - [2x1] vector containing Rx [azimuth; elevation] in degrees (AoA)
    %   txArray    - phased.URA object for transmit array configuration
    %   rxArray    - phased.ULA object for receive array configuration
    %   lambda     - Scalar wavelength in meters
    %
    % Outputs:
    %   v_tx       - [(MxN)x1] transmit steering vector for URA of size MxN
    %   v_rx       - [Px1] receive steering vector for ULA of size P
    %
    % Example:
    %   aod = [30; 45];  % Tx azimuth = 30°, elevation = 45°
    %   aoa = [60; 20];  % Rx azimuth = 60°, elevation = 20°
    %   txArray = phased.URA('Size', [4 4], 'ElementSpacing', [0.5 0.5]);
    %   rxArray = phased.ULA('NumElements', 8, 'ElementSpacing', 0.5);
    %   lambda = 0.1;  % 10cm wavelength
    %   [v_tx, v_rx] = steering_vec(aod, aoa, txArray, rxArray, lambda);

    aod_angles_rad = deg2rad(aod_angles);
    aoa_angles_rad = deg2rad(aoa_angles);

    if isa(txArray, 'phased.URA')
        v_tx = steering_vector_tx(aod_angles_rad, txArray.Size, txArray.ElementSpacing, lambda);
    else
        error('Transmit array must be a phased.URA object');
    end

    if isa(rxArray, 'phased.ULA')
        v_rx = steering_vector_rx(aoa_angles_rad, rxArray.NumElements, rxArray.ElementSpacing, lambda);
    else
        error('Receive array must be a phased.ULA object');
    end

end

function v = steering_vector_tx(angles, array_size, element_spacing, lambda)
    % STEERING_VECTOR_TX Compute steering vector for Uniform Rectangular Array
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
    az = angles(1);
    el = angles(2);

    indices = (0:num_elements - 1).';
    x_positions = (indices - (num_elements - 1) / 2) * element_spacing;

    % compute wavenumber (k = 2π/λ)
    k = 2 * pi / lambda;
    v = exp(-1j * k * (x_positions * cos(el) * cos(az)));
end