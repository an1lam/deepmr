# TF Cooperativity Simulation
Run all Python stuff:

    python tf_coop_main.py --seed 52  --train_sequences 10000 --test_sequences 1000 --variant_augmentation_percentage .25 --log_summary_stats --data_dir ../dat/sim_e2e --epochs 100 --n_conv_layers 3 --n_dense_layers 2 --seed 42 --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor.pt --model_fname cnn_counts_predictor.pt --model_type ensemble --n_reps 5 --n_rounds 50

This runs without either type of confounding. For sequence-dependent confounding, add `--confounder_motif SOX2_1`. For sequence-independent confounding, add `--confounder_prob .5`.

# DeepSEA
## Downloading necessary files
Reproducing the DeepSEA experiment from the paper requires downloading a combination of feature metadata and genomic data. First, let's switch to the `src/` directory (`cd src/`) and create a directory for storing all the DeepSEA files. I use `dat/deepsea` (relative to repo top-level) but everything that follows can easily be configured for other directory paths.
```{lang=sh}
mkdir -p ../dat/deepsea
export DEEPSEA_DATA_DIR="../dat/deepsea"
```

Second, we need to download a list of features predicted by DeepSEA with corresponding annotations of data source, cell types, feature type, and other features we don't care about.
```{lang=sh}
curl http://deepsea.princeton.edu/media/help/posproportion.txt > $DEEPSEA_DATA_DIR/deepsea_cols.tsv
```
We'll also download (manually, unfortunately) a list of URLs from the DeepSEA [Nature paper](https://www.nature.com/articles/nmeth.3547#Sec11) which point to BED files containing peaks for each feature DeepSEA predicts. To do this, download "Supplementary Table 1" and convert it to a TSV file.

Third, we're going to download a list of all human TFs from [here](http://humantfs.ccbr.utoronto.ca/download.php).
```{lang=sh}
curl http://humantfs.ccbr.utoronto.ca/download/v_1.01/DatabaseExtract_v_1.01.csv > $DEEPSEA_DATA_DIR/human_tfs.csv
```

Last, we're going to download the human reference genome (V19):


