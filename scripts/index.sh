INDEX_HOME=$1

# Get location of project source directory:
# line taken from https://stackoverflow.com/questions/59895/how-can-i-get-the-source-directory-of-a-bash-script-from-within-the-script-itsel/246128#246128
PROJECT_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

mkdir -p $INDEX_HOME

python $PROJECT_HOME/src/helper/prepare_clef.py --tgt_dir $INDEX_HOME

languages=( en de it fi ru )
for LANG in "${languages[@]}"
do

  echo Indexing $LANG

  mkdir -p $INDEX_HOME/$LANG/

  python -m pyserini.index \
    --input $INDEX_HOME/$LANG \
    --collection JsonCollection \
    --generator DefaultLuceneDocumentGenerator \
    --index $INDEX_HOME/$LANG/ \
    --threads 10 \
    --language $LANG \
    --storePositions --storeDocvectors --storeRaw
  
  echo Done
  
done
