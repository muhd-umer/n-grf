function pl = fspl(d, f)
    % FSPL Compute free-space path loss (FSPL)
    %
    % Description:
    %   Calculates the free-space path loss in dB for a given distance and frequency.
    %
    % Inputs:
    %   d - Distance between transmitter and receiver (meters)
    %   f - Frequency (Hz)
    %
    % Output:
    %   pl - Path loss in dB
    %
    % Example:
    %   pl = fspl(100, 2.4e9);

    pl = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/physconst("lightspeed"));
end