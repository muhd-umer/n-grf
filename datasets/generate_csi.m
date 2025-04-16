function [H, AoD, AoA] = generate_csi(rays, fc, cfg, txArray, rxArray, method, scenario, use_single_sc, sc_idx)
    % GENERATE_CSI Generate MIMO channel matrix using array steering vectors.
    %
    % This function computes a MIMO channel matrix H such that each TX-RX
    % pair receives a weighted contribution based on the ray's complex gain
    % and the spatial response of the arrays at the transmitter and receiver.
    %
    % Inputs:
    %   rays        - structure array containing ray information
    %   fc          - carrier frequency (Hz)
    %   cfg         - WLAN configuration (used for OFDM parameters) or NR Carrier Config
    %   txArray     - phased.URA object for transmit array
    %   rxArray     - phased.ULA object for receive array
    %   method      - channel computation method ("sbr" or alternative)
    %   scenario    - "indoor" or "outdoor"
    %   use_single_sc - boolean flag for single subcarrier processing
    %   sc_idx      - (optional) subcarrier index to use when use_single_sc is true
    %
    % Outputs:
    %   H   - Channel matrix of size (num_tx_ant, num_rx_ant, numSubcarriers)
    %   AoD - Matrix containing angles of departure (2 x numRays)
    %   AoA - Matrix containing angles of arrival (2 x numRays)

    if nargin < 8
        use_single_sc = false;
    end

    if nargin < 9 % Default sc_idx to empty if not provided
        sc_idx = [];
    end

    num_tx_ant = prod(txArray.Size);
    num_rx_ant = rxArray.NumElements;

    if scenario == "indoor"
        ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
        activeIndices = ofdmInfo.ActiveFrequencyIndices;
        sc_spacing = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;

        if use_single_sc

            if isempty(sc_idx)
                sc_idx = ceil(length(activeIndices) / 2);
            end

            freqs = fc + activeIndices(sc_idx) * sc_spacing;
            numSubcarriers = 1;
        else
            freqs = fc + activeIndices * sc_spacing;
            numSubcarriers = length(activeIndices);
        end

    elseif scenario == "outdoor"
        sc_spacing = cfg.SubcarrierSpacing * 1e3;
        numSubcarriersTotal = cfg.NSizeGrid * 12;
        activeIndices = (-numSubcarriersTotal / 2:numSubcarriersTotal / 2 - 1);

        if use_single_sc

            if isempty(sc_idx)
                zero_center_idx = find(activeIndices == 0);

                if isempty(zero_center_idx)
                    sc_idx = ceil(length(activeIndices) / 2);
                else
                    sc_idx = zero_center_idx;
                end

            end

            freqs = fc + activeIndices(sc_idx) * sc_spacing;
            numSubcarriers = 1;
        else
            freqs = fc + activeIndices * sc_spacing;
            numSubcarriers = length(freqs);
        end

    else
        error('Invalid scenario: use "indoor" or "outdoor"');
    end

    H = zeros(num_tx_ant, num_rx_ant, numSubcarriers);
    numRays = length(rays);
    AoD = zeros(2, numRays);
    AoA = zeros(2, numRays);

    lambda = physconst("lightspeed") / fc;

    for rayIdx = 1:numRays
        ray = rays(rayIdx);
        AoD(:, rayIdx) = ray.AngleOfDeparture;
        AoA(:, rayIdx) = ray.AngleOfArrival;
        
        % compute steering vectors
        [a_tx, a_rx] = steering_vec(ray.AngleOfDeparture, ray.AngleOfArrival, txArray, rxArray, lambda);

        for scIdx = 1:length(freqs)
            f = freqs(scIdx);

            if strcmp(method, "sbr")
                pl = ray.PathLoss;
                phase = ray.PhaseShift;
            else
                pl = fspl(ray.PropagationDistance, f);
                phase = 2 * pi * f * ray.PropagationDistance / physconst("lightspeed");
            end

            h = 10 ^ (-pl / 20) * exp(-1j * phase);
            H(:, :, scIdx) = H(:, :, scIdx) + h * (a_tx * a_rx.');
        end
    end
end
