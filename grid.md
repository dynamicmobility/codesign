
I want to refactor the sampling stages of each learning algorithm 
- src/codesign/hyperdesigners/mo_design_predictor_hypernetwork.py (DPUP)
- src/codesign/hyperdesigners/mo_design_hypernetwork.py (USUP)
- src/codesign/hyperdesigners/design_hypernetwork.py (DH)

In each algorithm, in some form, there exists a sampling algorithm that is grid based. In DH (single objective), designs are sampled for the same default tradeoff. In USUP (multi objective), designs and tradeoffs are sampled in a grid. In DPUP (multi objective), it is the same as USUP, except that the designs come from a prediction network d ~ f(w).

I want to make a unified grid class that can represent all of these sampling methods, and thus reduce code overall in the repository. The grid should look like so
```python
@dataclass
Grid:
    designs: np.ndarray # (M, K, n_d)
    tradeoffs: np.ndarray # (M, K, n_r)
    per_cell: int # per cell repeats of each design and tradeoff
    n_r: int # number of objectives
    rewards: (M, K, C, n_r)

    def batch_by_design(self) -> np.ndarray # (B grids with Grid.reward shape (M/B, K, C, n_r), where B divides M)
        # TODO: implement this
        pass

    def batch_by_tradeoff(self) -> np.ndarray # (B grids with Grid.reward shape (M, K / B, C, n_r), where B divides K)
        # TODO: implement this
        pass

    def shape(self):
        return self.rewards.shape

    @property
    def n_tradeoffs(self):
        pass

    @property
    def n_designs(self):
        pass
```

In this way, it should be possible to represent all sampling types in each algorithm. For instance, DH would be Grid.shape = (M, 1, C, 1), since it is single tradeoff, single objective. 

This grid should also be the thing that is saved in scripts/generate_data.py, and used all over the repository. It is a common dataset that all scripts and algorithms share. Moreover, there is currently a "sample grid" and a "rollout grid". However, each can be each other, for instance, a rollout grid is a sampled grid if the rewards array is None


Your tasks are
1. Identify where such a grid could be swapped in to each training algorithm listed above
2. Write the code for the grid, replacing the grids that currently exist in src/codesign/utils/grid.py.
3. Take the good parts of each of the grids that currently exist, like the different sampling strategies of DesignTradeoffSampleGrid, the from_predictor method of DesignPredictorSampleGrid (note in this case, the group size is the number of designs, M), the build_models, flatten, and unflatten functions, the saving and loading functions to npz, etc.


Please ask questions if you are unsure. Ask as many as you need.
