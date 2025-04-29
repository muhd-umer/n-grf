function [H, AoD, AoA] = generate_csi(rays, fc, cfg, txArray, rxArray, method, scenario, use_single_sc, sc_idx)
    % GENERATE_CSI Generate a spatially-consistent, frequency-selective MIMO channel
    %
    % Description:
    %   Generates a MIMO channel tensor H from ray tracing data.
    %   Computes the channel response for each subcarrier in the OFDM/NR signal.
    %   The channel tensor is computed using the steering vectors of the transmit
    %   and receive arrays, and the path loss and phase shift of each ray.
    %
    % Inputs:
    %   rays          - Ray objects array from ray tracing
    %   fc            - Center frequency (Hz)
    %   cfg           - OFDM/NR carrier configuration struct
    %   txArray       - Transmit array (phased.URA or phased.IsotropicAntennaElement)
    %   rxArray       - Receive array (phased.ULA or phased.IsotropicAntennaElement)
    %   method        - 'sbr' to use ray.PathLoss/PhaseShift, 'fspl' for pure FSPL
    %   scenario      - "indoor" or "outdoor"
    %   use_single_sc - true to pick one subcarrier via sc_idx, false for all
    %   sc_idx        - Subcarrier index if use_single_sc==true
    %
    % Outputs:
    %   H   - Nt x Nr x Nsc channel tensor (complex)
    %   AoD - 2 x Nrays matrix of departure [az;el] in degrees
    %   AoA - 2 x Nrays matrix of arrival   [az;el] in degrees
    %
    % Example:
    %   [H, AoD, AoA] = generate_csi(rays, fc, cfg, txArray, rxArray, 'sbr', "outdoor", true, 1);

    if nargin < 8, use_single_sc = false; end
    if nargin < 9, sc_idx = []; end

    is_siso = isa(txArray, 'phased.IsotropicAntennaElement') && isa(rxArray, 'phased.IsotropicAntennaElement');

    if scenario == "indoor"
        ofdmInfo   = wlanNonHTOFDMInfo('L-LTF', cfg.ChannelBandwidth);
        actIdx     = ofdmInfo.ActiveFrequencyIndices;
        sc_sp      = wlanSampleRate(cfg.ChannelBandwidth) / ofdmInfo.FFTLength;
        if use_single_sc
            if isempty(sc_idx), sc_idx = ceil(numel(actIdx)/2); end
            freqs = fc + actIdx(sc_idx)*sc_sp;
        else
            freqs = fc + actIdx*sc_sp;
        end
    else
        sc_sp      = cfg.SubcarrierSpacing * 1e3;
        totalSC    = cfg.NSizeGrid * 12;
        actIdx     = -totalSC/2 : totalSC/2-1;
        if use_single_sc
            if isempty(sc_idx), sc_idx = ceil(numel(actIdx)/2); end
            freqs = fc + actIdx(sc_idx)*sc_sp;
        else
            freqs = fc + actIdx*sc_sp;
        end
    end
    Nsc = numel(freqs);

    if is_siso
        Nt = 1;
        Nr = 1;
    else
        Nt = prod(txArray.Size);
        Nr = rxArray.NumElements;
    end
    H  = zeros(Nt, Nr, Nsc);

    if ~is_siso
        svTx = phased.SteeringVector( ...
            'SensorArray',            txArray, ...
            'PropagationSpeed',       physconst('LightSpeed'), ...
            'IncludeElementResponse', false );
        svRx = phased.SteeringVector( ...
            'SensorArray',            rxArray, ...
            'PropagationSpeed',       physconst('LightSpeed'), ...
            'IncludeElementResponse', false );
    end

    numRays = numel(rays);
    AoD     = zeros(2, numRays);
    AoA     = zeros(2, numRays);

    for r = 1:numRays
        ray  = rays(r);
        dist = ray.PropagationDistance;
        tau  = dist / physconst('LightSpeed');

        if ray.LineOfSight
            pts = [ray.TransmitterLocation, ray.ReceiverLocation];
        else
            pts = [ray.TransmitterLocation, ray.Interactions.Location, ray.ReceiverLocation];
        end

        % departure
        vecTx      = pts(:,2) - pts(:,1);
        [azT, elT] = cart2sph(vecTx(1), vecTx(2), vecTx(3));
        azT = rad2deg(azT);  elT = rad2deg(elT);
        AoD(:,r) = [azT; elT];

        % arrival
        vecRx      = pts(:,end) - pts(:,end-1);
        [azR, elR] = cart2sph(vecRx(1), vecRx(2), vecRx(3));
        azR = rad2deg(azR);  elR = rad2deg(elR);
        AoA(:,r) = [azR; elR];

        if ~is_siso
            aTx = svTx(fc, [azT; elT]);
            aRx = svRx(fc, [azR; elR]);
            array_gain = aTx * aRx.';
        else
            array_gain = 1; % isotropic antennas have gain of 1 (0 dBi)
        end

        for k = 1:Nsc
            f = freqs(k);
            if strcmp(method, 'sbr')
                pl0   = ray.PathLoss;
                phi0  = ray.PhaseShift;
                pl    = pl0;
                phase = phi0 + 2*pi*(f - fc)*tau;
            else
                pl    = fspl(dist, f);
                phase = 2*pi*f*tau;
            end
            h = 10^(-pl/20) * exp(-1j*phase);
            H(:,:,k) = H(:,:,k) + array_gain * h;
        end
    end
end
