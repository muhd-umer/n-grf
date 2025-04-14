# baselines/mlp.py
# TODO: Create an MLP that estimates the MIMO channes
# Understand the dataloader, i.e., what the target channel matrix is per batch
# Then, create an MLP network that takes as input the receiver position (any
# other features) and "maps" to the channel matrix with the goal of minimizing
# the NMSE/any other loss function Regardless of the loss function, the SNR
# values should be tracked and will serve as the main comparison metric against
# other approaches SNR = -10 * log10(NMSE), where NMSE = ||H - H_hat||^2 /
# ||H||^2
