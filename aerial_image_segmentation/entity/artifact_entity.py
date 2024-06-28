from dataclasses import dataclass
from torch.utils.data.dataloader import DataLoader

@dataclass
class DataIngestionArtifact:
     zip_file_path : str