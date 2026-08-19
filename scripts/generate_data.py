import codesign

# TODO: add a design predictor only flag
# TODO: make it so that when only m tradeoffs are specified it reduces to corners of simplex
# TODO: add in a loading function to the npz
# TODO: add an easy way to check what keys exist in codesign.MOCOPredictorGrid.data

def main(args):

    # Open mo design predictor hypernetwork using args.model_path

    # Use the GPU to rollout a design tradeoff grid. There should be
    # num tradeoffs, num designs, and num_per_cell (use codesign.rollout_mo_design_hypernetwork)

    # Evaluate the design predictor across NUM_TRADEOFFs

    # Store into a 
    codesign.MOCOPredictorGrid

    # Save this as an npz at args.save_path

if __name__ == '__main__':
    # parse args
    main(args)