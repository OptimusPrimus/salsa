#!/usr/bin/bash

wget https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD/download/passt_roberta.1.ckpt
wget https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD/download/passt_roberta.2.ckpt
wget https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD/download/passt_roberta.3.ckpt
wget https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD/download/passt_roberta.4.ckpt

cat passt_roberta.*.ckpt > passt_roberta.ckpt
rm passt_roberta.*.ckpt