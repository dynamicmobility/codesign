import codesign
import matplotlib.pyplot as plt
import numpy as np
import moplayground as mop

def main():
    fig, axs = plt.subplots(ncols=3)
    axs = axs.flatten()
    dpup = codesign.Grid.load('scripts/outputs/datasets/7sesooah/sweep.npz')
    usup = codesign.Grid.load('scripts/outputs/datasets/ls8xn17y/sweep.npz')
    for i, ax in enumerate(axs):
        value_ax = ax.twinx()
        for dataset, color, label in (
            (dpup, 'C0', 'DPUP value prediction'),
            (usup, 'C1', 'USUP value prediction'),
        ):
            # column i is tradeoff i: its (n_designs, per_cell) values and own designs
            values = np.asarray(dataset.data['value'][:, i]).mean(axis=1)
            designs = dataset.designs[:, i, 0]
            order = np.argsort(designs)
            value_ax.plot(
                designs[order], values[order], '--', color=color, label=label,
            )

        codesign.plot_design_sweep_1d(
            ax            = ax,
            designs       = dpup.designs[:, i],
            returns       = dpup.rewards[:, i] @ dpup.unique_tradeoffs[i],
            style         = 'band',
            bins          = 32,
            sweep_label   = 'DPUP',
            optimum_label = 'DPUP optimal design'
        )

        codesign.plot_design_sweep_1d(
            ax                  = ax,
            designs             = usup.designs[:, i],
            returns             = usup.rewards[:, i] @ usup.unique_tradeoffs[i],
            style               = 'band',
            bins                = 32,
            sweep_color         = 'C1',
            best_point_color    = 'C1',
            sweep_label         = 'USUP',
            optimum_label       = 'USUP optimal design',
            points_label        = None,
            band_label          = None
        )
        value_ax.set_ylabel('predicted value')
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            value_handles, value_labels = value_ax.get_legend_handles_labels()
            ax.legend(handles + value_handles, labels + value_labels)
        ax.set_title(f'Tradeoff = {dpup.unique_tradeoffs[i]}')
    
    # fig.legend()
    fig.set_size_inches((15, 8))
    fig.tight_layout()
    fig.savefig('scripts/outputs/dsup_vs_usup.svg')

if __name__ == '__main__':
    main()
