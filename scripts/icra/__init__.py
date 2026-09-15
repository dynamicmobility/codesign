from dataclasses import dataclass
from pathlib import Path

@dataclass
class Run:
    run_id: str
    dataset_path: Path
    checkpoint: int

MO_CHECKPOINT = 3e9

FINAL_CONFIGS: dict[str, dict[str, Run]] = {
    'cheetah': {
        'MDH': Run(
            run_id          = '5c1e0l9u',
            dataset_path    = 'scripts/icra/outputs/',
            checkpoint      = MO_CHECKPOINT
        ),
        'MLP': Run(
            run_id          = 'dkfq4g7t',
            dataset_path    = 'scripts/icra/outputs/',
            checkpoint      = MO_CHECKPOINT
        ),
        # 'MORLAX': '',
        'PPO': [
            # Run(run_id='pa84drhm', dataset_path=None, checkpoint=-1),
            # Run(run_id='7l538fam', dataset_path=None, checkpoint=-1),
            # Run(run_id='j5slo6nk', dataset_path=None, checkpoint=-1),
            # Run(run_id='krmd1b3t', dataset_path=None, checkpoint=-1),
            # Run(run_id='4mo9odyn', dataset_path=None, checkpoint=-1),
            # Run(run_id='8gt6dx33', dataset_path=None, checkpoint=-1),
        ]
    },
    'walker': {
        'MDH': Run(
            run_id          = 'nlpnls3l',
            dataset_path    = 'scripts/icra/outputs/',
            checkpoint      = MO_CHECKPOINT
        ),
        'MLP': Run(
            run_id          = 'zbvakupw',
            dataset_path    = 'scripts/icra/outputs/',
            checkpoint      = MO_CHECKPOINT
        ),
        # 'MORLAX': '',
        'PPO': ['']
    }
}


DATA_GEN = {
    'cheetah': {
        'pop_size'    : 1024,
        'budget'      : 32768,
        'workers'     : 1
    },
    'walker': {
        'pop_size'    : 1024,
        'budget'      : 32768,
        'workers'     : 1
    }
}
