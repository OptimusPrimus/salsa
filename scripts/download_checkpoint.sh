#!/usr/bin/bash

wget https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD/download/xaa
wget https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD/download/xab
wget https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD/download/xac
wget https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD/download/xad
wget https://cloud.cp.jku.at/index.php/s/pMWRbJzXqFPPzgD/download/xae

cat xa* > passt_roberta.ckpt
rm passt_roberta.*.ckpt