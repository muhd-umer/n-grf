function [H, AoD, AoA] = generate_csi(rays, fc, cfg, num_tx_ant, num_rx_ant, method, scenario, use_single_sc, sc_idx)

    if nargin < 8
        use_single_sc = false;
    end

    if scenario == "indoor"
        ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
        activeIndices = ofdmInfo.ActiveFrequencyIndices;
        sc_spacing = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;

        if use_single_sc

            if nargin < 9 || isempty(sc_idx)
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
        numSubcarriers = cfg.NSizeGrid * 12;
        activeIndices = (-numSubcarriers / 2:numSubcarriers / 2 - 1);

        if use_single_sc

            if nargin < 9 || isempty(sc_idx)
                sc_idx = ceil(length(activeIndices) / 2);
            end

            freqs = fc + activeIndices(sc_idx) * sc_spacing;
            numSubcarriers = 1;
        else
            freqs = fc + activeIndices * sc_spacing;
        end

    else
        error('Invalid scenario: use "indoor" or "outdoor"');
    end

    H = zeros(num_tx_ant, num_rx_ant, numSubcarriers);
    numRays = length(rays);
    AoD = zeros(2, numRays);
    AoA = zeros(2, numRays);

    for rayIdx = 1:numRays
        ray = rays(rayIdx);
        AoD(:, rayIdx) = ray.AngleOfDeparture;
        AoA(:, rayIdx) = ray.AngleOfArrival;

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

            for rx = 1:num_rx_ant

                for tx = 1:num_tx_ant
                    H(tx, rx, scIdx) = H(tx, rx, scIdx) + h;
                end
            end
        end
    end
end
