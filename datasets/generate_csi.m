function [H, AoD, AoA] = generate_csi(rays, fc, cfg, num_tx_ant, num_rx_ant, method, scenario)

    if scenario == "indoor"
        ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
        sc_spacing = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;
        freqs = fc + ofdmInfo.ActiveFrequencyIndices * sc_spacing;
        numSubcarriers = length(ofdmInfo.ActiveFrequencyIndices);
    elseif scenario == "outdoor"
        sc_spacing = cfg.SubcarrierSpacing * 1e3;
        numSubcarriers = cfg.NSizeGrid * 12;
        activeFreqIndices = (-numSubcarriers / 2:numSubcarriers / 2 - 1);
        freqs = fc + activeFreqIndices * sc_spacing;
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
