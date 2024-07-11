#!/usr/bin/bash

wget https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD/download/passt_roberta.1.ckpt
wget https://cloud.cp.jku.at/index.php/s/aMDLmLTZH4jCHwq/download/passt_roberta.2.ckpt
wget https://cloud.cp.jku.at/index.php/s/3kKiQJFk45R7S5k/download/passt_roberta.3.ckpt
wget https://cloud.cp.jku.at/index.php/s/ijLwzPAQrqKRf5P/download/passt_roberta.4.ckpt
wget https://cloud.cp.jku.at/index.php/s/Np4A5nobAkaSGpY/download/passt_roberta.5.ckpt

cat passt_roberta.*.ckpt > passt_roberta.ckpt
# rm passt_roberta.*.ckpt

#https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD
#https://cloud.cp.jku.at/index.php/s/aMDLmLTZH4jCHwq
#https://cloud.cp.jku.at/index.php/s/3kKiQJFk45R7S5k
#https://cloud.cp.jku.at/index.php/s/ijLwzPAQrqKRf5P
#https://cloud.cp.jku.at/index.php/s/Np4A5nobAkaSGpY