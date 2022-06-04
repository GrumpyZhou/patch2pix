# Download and extracting validation set
# Dataset link: https://www.cs.ubc.ca/~kmyi/imw2020/data.html

mkdir immatch_benchmark
mkdir immatch_benchmark/val_dense
cd immatch_benchmark/val_dense

wget https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/reichstag.tar.gz
tar -xvzf reichstag.tar.gz

wget https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/sacre_coeur.tar.gz
tar -xvzf sacre_coeur.tar.gz

wget https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/st_peters_square.tar.gz
tar -xvzf st_peters_square.tar.gz

wget https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/taj_mahal.tar.gz
tar -xvzf taj_mahal.tar.gz

wget https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/temple_nara_japan.tar.gz
tar -xvzf temple_nara_japan.tar.gz

rm *.tar.gz
