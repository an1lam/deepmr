## TF Cooperativity Simulation
In our paper we discussed four simulation scenarios: no confounding, sequence-dependent confounding, sequence-independent confounding, and both types of confounding. The following command will simulate data for the first scenario using the settings we used for the paper.

```{lang=sh}
python tf_coop_main.py --train_sequences 10000 --test_sequences 1000 --variant_augmentation_percentage .25 --log_summary_stats --data_dir ../dat/sim_e2e --epochs 100 --n_conv_layers 3 --n_dense_layers 2 --seed 42 --train_data_fnames train_labels.csv --train_data_fnames train_variant_labels.csv --model_fname cnn_counts_predictor.pt --model_fname cnn_counts_predictor.pt --model_type ensemble --n_reps 5 --n_rounds 50
```

As noted, this runs without either type of confounding. For sequence-dependent confounding, add `--confounder_motif SOX2_1`. For sequence-independent confounding, add `--confounder_prob .5`. For both, just add both!

## BPNet experiments
