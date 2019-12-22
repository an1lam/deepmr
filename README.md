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

## Using JupyterLab on a GCE box
On a new box, there are a few things you have to do to get Jupyter working in a way that's compatible with this project's code.

1. As unix user 'jupyter', change `/home/jupyter/.jupyter/jupyter_notebook_config.py`'s `c.NotebookApp.notebook_dir` folder to `/home/<your username>/project/`. Once you've done this, restart Jupyter with `sudo service jupyter restart`.
2. Make it so that the 'jupyter' user can access files in the 'dat' dir by running `./scr/give_ipython_data_access.sh` on the box you're using (from directory `~/project/`).
3. Deal with the fact that `samtools` is a needy package that requires system-level dependencies.
    * Run `sudo apt-get install libbz2-dev liblzma-dev` as advised [here](https://samtools.github.io/bcftools/howtos/install.html).
    * Run `sudo apt-get install libcurl4-openssl-dev libssl-dev` (from [this SO post](https://stackoverflow.com/questions/11471690/curl-h-no-such-file-or-directory/11471743) and [this AskUbuntu post](https://askubuntu.com/questions/133806/getting-an-error-when-using-make-command-installing-aircrack-ng-on-ubuntu-12)).
      _This avoids the following and one other error_:
      ```
      htslib/hfile_libcurl.c:45:23: fatal error: curl/curl.h: No such file or directory
      #include <curl/curl.h>
                            ^
      compilation terminated.
      error: command 'gcc' failed with exit status 1
      ```

