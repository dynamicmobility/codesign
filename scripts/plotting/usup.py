import codesign
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop

def main():
    fig, axs = plt.subplots(ncols=3)
    axs = axs.flatten()
    dpup = codesign.DesignTradeoffDataset.load('scripts/outputs/datasets/7sesooah/sweep.npz')
    usup = codesign.DesignTradeoffDataset.load('scripts/outputs/datasets/ls8xn17y/sweep.npz')
    for i, ax in enumerate(axs):
        codesign.plot_design_sweep_1d(
            ax            = ax,
            designs       = dpup.designs,
            returns       = dpup.rewards[:, i] @ dpup.tradeoffs[i],
            style         = 'band',
            bins          = 32,
            sweep_label   = 'DPUP',
            optimum_label = 'DPUP optimal design'
        )

        codesign.plot_design_sweep_1d(
            ax                  = ax,
            designs             = usup.designs,
            returns             = usup.rewards[:, i] @ usup.tradeoffs[i],
            style               = 'band',
            bins                = 32,
            sweep_color         = 'C1',
            best_point_color    = 'C1',
            sweep_label         = 'USUP',
            optimum_label       = 'USUP optimal design',
            points_label        = None,
            band_label          = None
        )
        if i == 0:
            ax.legend()
        ax.set_title(f'Tradeoff = {dpup.tradeoffs[i]}')
    
    # fig.legend()
    fig.set_size_inches((15, 8))
    fig.tight_layout()
    fig.savefig('scripts/outputs/dsup_vs_usup.svg')

if __name__ == '__main__':
    main()
