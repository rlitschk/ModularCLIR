RESOURCES_DIR=$1

[[ -z "$1" ]] && { echo "Please specify positional parameter \$RESOURCES_DIR" ; exit 1; }

ADAPTER_DIR=$RESOURCES_DIR/adapter
echo Creating $ADAPTER_DIR
mkdir -p $ADAPTER_DIR

SFT_DIR=$RESOURCES_DIR/sft
echo Creating $SFT_DIR
mkdir -p $SFT_DIR

echo Downloading Language Adapters
wget https://madata.bib.uni-mannheim.de/391/4/language_adapters.tar.gz
tar -xzvf language_adapters.tar.gz -C $ADAPTER_DIR

echo Downloading Language Masks
wget https://madata.bib.uni-mannheim.de/391/6/language_masks.tar.gz
tar -xzvf language_masks.tar.gz -C $SFT_DIR

echo Downloading Ranking Adapters
wget https://madata.bib.uni-mannheim.de/391/3/ranking_adapters.tar.gz
tar -xzvf ranking_adapters.tar.gz -C $ADAPTER_DIR

echo Downloading Ranking Masks
wget https://madata.bib.uni-mannheim.de/391/5/ranking_masks.tar.gz
tar -xzvf ranking_masks.tar.gz -C $SFT_DIR

echo Downloading Monobert
wget https://madata.bib.uni-mannheim.de/391/2/monobert.tar.gz
tar -xzvf monobert.tar.gz -C $RESOURCES_DIR/

echo Downloading Pre-ranking files
wget https://madata.bib.uni-mannheim.de/391/1/preranking.tar.gz
tar -xzvf preranking.tar.gz -C $RESOURCES_DIR/

mkdir $RESOURCES_DIR/downloaded_files
mv $RESOURCES_DIR/*.tar.gz $RESOURCES_DIR/downloaded_files
