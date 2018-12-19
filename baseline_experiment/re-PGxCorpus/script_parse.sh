#parsing using the charniak parser (https://github.com/BLLIP/bllip-parser) +
#http://bllip.cs.brown.edu/download/bioparsingmodel-rel1.tar.gz
#python ModelFetcher.py -i GENIA+PubMed

data=`pwd`/$1
cd bllip-parser
sed 's/\(.*\)/<s> \1 <\/s>/' $data/$2 | first-stage/PARSE/parseIt -K -l399 -N50 $HOME/.local/share/bllipparser/GENIA+PubMed/parser/ | second-stage/programs/features/best-parses -l $HOME/.local/share/bllipparser/GENIA+PubMed/reranker/features.gz $HOME/.local/share/bllipparser/GENIA+PubMed/reranker/weights.gz > $data/$2.McClosky.trees
cd -
#cd ~/Bureau/loria/soft/
java -cp nlp4j-ddr.jar edu.emory.mathcs.nlp.bin.DDGConvert -i $data/$2.McClosky.trees
#cd -

th extractTrees.lua $data/$2.McClosky.trees.ddg
sed -i 's/-LRB-/(/g' $data/$2.McClosky.trees.ddg_tree_sentences
sed -i 's/-RRB-/)/g' $data/$2.McClosky.trees.ddg_tree_sentences
sed -i 's/-LSB-/[/g' $data/$2.McClosky.trees.ddg_tree_sentences
sed -i 's/-RSB-/]/g' $data/$2.McClosky.trees.ddg_tree_sentences
sed -i 's/-LCB-/{/g' $data/$2.McClosky.trees.ddg_tree_sentences
sed -i 's/-RCB-/}/g' $data/$2.McClosky.trees.ddg_tree_sentences
