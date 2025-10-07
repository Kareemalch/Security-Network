from dataclasses import dataclass

@dataclass
class DataingestionArtifact:
    train_file_path: str
    test_file_path: str