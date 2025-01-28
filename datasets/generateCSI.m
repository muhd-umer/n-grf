function H = generateCSI(rays, fc, cfg, num_tx_ant, num_rx_ant)
    ofdmInfo = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
    sc_spacing = wlanSampleRate(cfg.ChannelBandwidth)/ofdmInfo.FFTLength;
    freqs = fc + ofdmInfo.ActiveFrequencyIndices*sc_spacing;
    
    H = zeros(num_tx_ant, num_rx_ant, length(freqs));
    
    for scIdx = 1:length(freqs)
        for rayIdx = 1:length(rays)
            ray = rays(rayIdx);
            f = freqs(scIdx);
            
            % path loss and phase
            pl = fspl(ray.PropagationDistance, f);
            % disp("Path loss: " + pl + " dB");
            % disp("Path loss: " + ray.PathLoss + " dB");
            % assert(pl==ray.PathLoss)
            phase = 2*pi*f*ray.PropagationDistance/physconst("lightspeed");
            
            % complex coeffs
            h = 10^(-pl/20)*exp(-1j*phase);
            for rx = 1:num_rx_ant % for each pair
                for tx = 1:num_tx_ant
                    H(tx,rx,scIdx) = H(tx,rx,scIdx) + h;
                end
            end
        end
    end
end