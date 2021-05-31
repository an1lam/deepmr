# TF Cooperativity Simulation
## No Confounding
Generate simulated data:

    python tf_coop_simulation.py --seed 42  --train_sequences 100000 --test_sequences 10000 --variant_augmentation_percentage .25 --log_summary_stats

which should produce the following output:

    INFO:root:Training count summary stats: 
	exposure mean = 54.44, variance = 2492.82 
	outcome mean = 28.44, variance = 2015.25
    INFO:root:Label counts: 
	with exposure: 49905, with outcome: 49797, with both: 24907, with confounder motif: 0

Train no variant data model:

    python tf_coop_model.py --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed 42 --train_data_fnames train_labels.csv --model_fname cnn_counts_predictor_no_variants.pt

Train variant data model:

    python tf_coop_model.py --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed 42 --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor_no_variants.pt
    # Ensemble version
    for i in `seq 1 5`; do mkdir -p ../dat/sim/ensemble/$i/;    python tf_coop_model.py --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed $i --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor_with_variants.pt --data_dir ../dat/sim/ --model_fname ensemble/$i/cnn_counts_predictor.pt; done

### E2E version
Run all Python stuff:

    python tf_coop_main.py --seed 42  --train_sequences 100000 --test_sequences 10000 --variant_augmentation_percentage .25 --log_summary_stats --data_dir ../dat/sim_e2e --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed 42 --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor.pt --model_fname cnn_counts_predictor.pt --model_type ensemble --n_reps 10 

Updated:

    python tf_coop_main.py --seed 42  --train_sequences 10000 --test_sequences 1000 --variant_augmentation_percentage .25 --log_summary_stats --data_dir ../dat/sim_e2e --epochs 50 --n_conv_layers 2 --n_dense_layers 1 --seed 42 --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor.pt --model_fname cnn_counts_predictor.pt --model_type ensemble --n_reps 2 --n_rounds 1

## Local Confounding
### Sequence Based
#### E2E version
Run all Python stuff:

    python tf_coop_main.py --seed 42  --train_sequences 100000 --test_sequences 10000 --variant_augmentation_percentage .25 --log_summary_stats --data_dir ../dat/sim_e2e_conf --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed 42 --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor.pt --model_fname cnn_counts_predictor.pt --model_type ensemble --n_reps 10 --confounder_motif SOX2_1

which should produce the following output in the simulation step:

    INFO:root:Training count summary stats: 
	exposure mean = 72.26, variance = 2521.20 
	outcome mean = 55.84, variance = 2056.72
INFO:root:Label counts: 
	with exposure: 49691, with outcome: 50122, with both: 24938, with confounder motif: 75146

#### Step-by-step
Generate simulated data:

    python tf_coop_simulation.py --seed 42 --log_summary_stats --train_sequences 100000 --test_sequences 10000 --confounder_motif SOX2_1 --data_dir ../dat/sim_conf/ --variant_augmentation_percentage .25

which should produce the following output:

    INFO:root:Training count summary stats: 
	exposure mean = 87.01, variance = 2200.98 
	outcome mean = 73.99, variance = 2611.37
    INFO:root:Label counts: 
	with exposure: 66542, with outcome: 66732, with both: 44466, with confounder: 66651


Train no variant data model:

    python tf_coop_model.py --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed 42 --train_data_fnames train_labels.csv --model_fname cnn_counts_predictor_no_variants.pt --data_dir sim_conf

Train variant data model:

    python tf_coop_model.py --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed 42 --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor_no_variants.pt --data_dir ../dat/sim_conf
    # Ensemble version
    for i in `seq 1 5`; do mkdir -p ../dat/sim_conf/ensemble/$i/;    python tf_coop_model.py --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed $i --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor_with_variants.pt --data_dir ../dat/sim_conf/ --model_fname ensemble/$i/cnn_counts_predictor.pt; done

### Non-sequence Based
Generate simulated data:

    python tf_coop_simulation.py --seed 42 --log_summary_stats --train_sequences 100000 --test_sequences 10000 --confounder_prob .5 --data_dir ../dat/sim_conf_non_seq/ --variant_augmentation_percentage .25

which should produce the following output:

    INFO:root:Training count summary stats: 
    	exposure mean = 66.92, variance = 2656.22 
    	outcome mean = 40.98, variance = 2185.36
    INFO:root:Label counts: 
    	with exposure: 49905, with outcome: 49797, with both: 24907, with confounder motif: 0

Train variant data model:

    python tf_coop_model.py --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed 42 --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor_no_variants.pt --data_dir ../dat/sim_conf_non_seq
    # Ensemble version
    for i in `seq 1 5`; do mkdir -p ../dat/sim_conf_non_seq/ensemble/$i/;    python tf_coop_model.py --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed $i --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor_with_variants.pt --data_dir ../dat/sim_conf_non_seq/ --model_fname ensemble/$i/cnn_counts_predictor.pt; done

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


