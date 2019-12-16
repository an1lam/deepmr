# Gregor: Mendelian Randomization for Neural Nets
Source code for "Determining Causal Effects of Transcription Factor Binding on Chromatin Accessibility with Mendelian Randomization".

# Usage
## Environment Setup
Set up a Python 3 virtualenv in the '.venv/' directory in 'src/'.

## Recompiling Cython
This assumes that you have a virtual environment setup with the requisite dependencies.

Run `scr/compile_cython.sh` from the project base directory.

## Data Setup
_Placeholder note_: Currently used chromatin accessibility data comes from [here](https://www.encodeproject.org/experiments/ENCSR136DNA/) but has been stripped of all columns but the first 5.
