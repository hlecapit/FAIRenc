The easiest way to start is to use the command uv to manage and install python packages. The current project requires python 3.10.
Together with uv, use the pyproject et uv.lock files to run it. Do not forget to pin the correct python version.

Usage : 

uv run train_fairenc.py --dataset german

You can also change the seed for robust estimation, eg. :

uv run train_fairenc.py --dataset german --seed 45

