import pypolar as plr
import codesign
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import qUpperConfidenceBound, qLogExpectedImprovement
from functools import partial
from scipy.stats import gaussian_kde

OBJ     = 2                 # objective the one-hot tradeoff picks out
BETA    = 2.0               # qUCB exploration weight
Q       = 100               # designs proposed per generation
SEED    = 0                 # optimize_acqf restarts and the qUCB sampler draw from this
BOUNDS  = (0.5, 2.0)        # bthigh scale, from config/mo_design_hypernetwork/cheetah1D.yaml
GRID    = 2048              # actions the posterior is drawn over
PATHS   = 20
DATASET = 'scripts/outputs/datasets/ls8xn17y/sweep.npz'
PROPOSAL_COLOR = '#0072B2'


def main():
    torch.manual_seed(SEED)

    usup  = codesign.DesignTradeoffDataset.load(DATASET)
    names = codesign.utils.plotting.objective_labels(usup.objectives)
    
    objectives = plr.DecoupledObjectives(
        objectives = [
            plr.Objective.from_data(
                actions       = usup.designs,
                values        = usup.rewards[:, j, 0, j],
                maximize      = True,
                name          = name,
                action_bounds = BOUNDS
            )
            for j, name in enumerate(names)
        ]
    )

    mogp = plr.DecoupledMOGP(
        objectives          = objectives,
        noise               = plr.NoiseModel.fitted(),
        fit_hyperparameters = True
    )

    w  = np.eye(len(objectives))[OBJ]
    gp = mogp.scalarized(w)

    acq = plr.AcquisitionFunction(
        # acqf      = partial(qUpperConfidenceBound, beta=BETA),
        acqf = partial(qLogExpectedImprovement),
        objective = objectives,
        raw_samples=2048
    )
    generation = np.sort(acq.query(gp, q=Q)[:, 0])

    print(f'tradeoff {w}, beta {BETA}, {Q} designs')
    print('  ' + ' '.join(f'{d:.3f}' for d in generation))

    X = np.linspace(*BOUNDS, num=GRID).reshape((-1, 1))
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 9))
    axs = axs.flatten()

    # each objective in its own units, so the three panels are readable
    mu, std = mogp.posterior_at(X, raw=True)
    paths   = mogp.sample_paths(X, num_paths=PATHS, raw=True)
    for j, (ax, objective) in enumerate(zip(axs, objectives.objectives)):
        plr.plot_fit_1d(
            ax    = ax,
            x     = X,
            mu    = mu[:, j],
            std   = std[:, j],
            xdata = objective.xdata,
            ydata = objective.ydata,
            paths = paths[:, :, j],
            title = objective.name
        )

    mu_w, std_w = gp.posterior_at(X)
    ax = plr.plot_fit_1d(
        ax    = axs[-1],
        x     = X,
        mu    = mu_w[:, 0],
        std   = std_w[:, 0],
        xdata = objectives.objectives[0].xdata,
        ydata = np.stack([o.standard_y for o in objectives.objectives], axis=1) @ w,
        paths = gp.sample_paths(X, num_paths=PATHS)[:, :, 0],
        title = f'scalarized, w = {w}'
    )
    ax.set_ylabel('standardized objective')
    for i, design in enumerate(generation):
        ax.axvline(design, color=PROPOSAL_COLOR, lw=0.5, ls='--', alpha=0.22,
                   zorder=5, label=f'{Q} proposed designs' if i == 0 else None)

    # a density over proposals sits in probability, not in the objective's
    # units, so it needs its own axis
    density = ax.twinx()
    density.fill_between(
        X[:, 0], gaussian_kde(generation)(X[:, 0]), color=PROPOSAL_COLOR,
        alpha=0.22, lw=0, label='proposal density'
    )
    density.set_ylabel('proposal density')
    density.set_ylim(bottom=0)

    # a twinned axes paints over the whole original, so zorder alone cannot put
    # the density behind the fit; the axes themselves have to be reordered
    density.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    handles = ax.get_legend_handles_labels()
    extra   = density.get_legend_handles_labels()
    ax.legend(handles[0] + extra[0], handles[1] + extra[1], frameon=False,
              fontsize=8, loc='lower right')

    fig.tight_layout()
    fig.savefig('test.svg')


if __name__ == '__main__':
    main()
