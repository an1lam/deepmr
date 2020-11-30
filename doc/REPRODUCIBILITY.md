
# TF Cooperativity Simulation
Generate simulated data:

    python tf_coop_simulation.py --seed 42  --train_sequences 100000 --test_sequences 10000 --variant_augmentation_percentage .25 --log_summary_stats

Train no variant data model:

    python tf_coop_model.py --epochs 50 --n_conv_layers 3 --n_dense_layers 3 --seed 42 --train_data_fnames train_labels.csv --model_fname cnn_counts_predictor_no_variants.pt
