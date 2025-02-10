from datasets.dataloader import get_wireless_dataloader

dataloader = get_wireless_dataloader(
    "path/to/data.mat", batch_size=32, num_pc=1000  # Sample 1000 points randomly
)

for batch in dataloader:
    point_cloud = batch["point_cloud"]  # [B, N, 3]
    tx_position = batch["tx_position"]  # [B, 3]
    rx_position = batch["rx_position"]  # [B, 3]
    channel_matrix = batch["channel_matrix"]  # [B, num_tx, num_rx, num_subcarriers]
    # ... process batch
