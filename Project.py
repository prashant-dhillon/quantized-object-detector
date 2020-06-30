from dataclasses import dataclass
from pathlib import Path


@dataclass
class Project:
    """
    This class represents our project. It stores useful information about the structure, e.g. patchs.
    """
    base_dir: Path = Path(__file__).parents[0]
    val_data_dir = base_dir / 'dataset'  / 'VOC' / '2007' / 'test'
    train_data_dir =  base_dir / 'dataset' / 'VOC' / '2007' / 'train'
    checkpoint_dir = base_dir / 'checkpoint'
    trained_model_dir = base_dir / 'trained_models'
    quantized_trained_model_dir = trained_model_dir / 'quantized'
    pruned_model_dir = trained_model_dir / 'pruned'
    eval_results_dir = base_dir / 'eval_results'
    model_temp_dir = Path('/tmp')
    model_temp_dir = Path('/tmp')

    def __post_init__(self):
        # create the directory if they does not exist
        self.val_data_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.trained_model_dir.mkdir(exist_ok=True)
        self.pruned_model_dir.mkdir(exist_ok=True)
        self.eval_results_dir.mkdir(exist_ok=True)


# expose a singleton
project = Project()
