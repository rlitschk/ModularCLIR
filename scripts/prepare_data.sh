PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Specify output directory for downloading and storing prepared data files.
OUTPUT_DIR=/home/usr/resource/data/msmarco
mkdir -p $OUTPUT_DIR

pytohn $PROJECT_HOME/src/util/prepare_msmarco.py --output_dir $OUTPUT_DIR
