# Figures and Experiments for the Bio-Temporal-Dynamics MDPI Paper

Please place notebooks for generating figures in the `figures` folder.
You can include the `figure_utils.py` which automatically causes your matplotlib figures to have the same style as the ones used in Andreas' PhD thesis.
This script also provides some useful helper functions and colour definitions.

However, for this script to work, make sure that
* You have a recent version of GhostScript installed (the `gs` executable must be in your PATH); GhostScript is required to crop all figures to their bounding box (because Matplotlib on its own sucks at doing that corectly) and to reduce the generated PDF file size (Matplotlib sucks at that too).
* Make sure that you have an installation of TeXlive in your PATH that provides `siunitx.sty`, `libertine.sty`, `libertinust1math.sty`, `mathrsfs.sty` and `amssymb.sty`


## Dependencies

The notebook `probability_tuning_curve.ipynb` depends on the git repository `https://github.com/ctn-waterloo/ssp-bayesopt`. Please install before running that notebook.
