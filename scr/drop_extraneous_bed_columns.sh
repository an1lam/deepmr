find $PROJECT_DIR/dat/ -name *.bed.gz -exec .venv/bin/python data/drop_extraneous_bed_columns.py 3 {} \;
