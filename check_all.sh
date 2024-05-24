#!/bin/bash
set -Ceux

# Check all commands working
cd deploy/fluxgnn_icml2024

cp ../../data/datasets.tar.gz data
cd data && tar xvf datasets.tar.gz
cd ..
make install

l1=$( \
  qexe -c 1 -m 32 \
  make mixture_fluxgnn_eval \
  | grep Log | cut -d ' ' -f3)
l2=$( \
  qexe -c 1 -m 32 \
  make mixture_penn_eval \
  | grep Log | cut -d ' ' -f3)
l3=$( \
  qexe -c 1 -m 32 \
  make mixture_mppde_eval \
  | grep Log | cut -d ' ' -f3)

l4=$( \
  qexe -c 1 -m 32 \
  make cd_fluxgnn_eval \
  | grep Log | cut -d ' ' -f3)
l5=$( \
  qexe -c 1 -m 32 \
  make cd_fvm_eval \
  | grep Log | cut -d ' ' -f3)


l6=$( \
  qexe -c 1 -m 32 \
  make test_cd_fluxgnn_train \
  | grep Log | cut -d ' ' -f3)
l7=$( \
  qexe -c 1 -m 32 \
  make test_mixture_fluxgnn_train \
  | grep Log | cut -d ' ' -f3)

l8=$( \
  qexe -c 1 -m 32 -g 1 \
  make mixture_penn_train \
  | grep Log | cut -d ' ' -f3)
l9=$( \
  qexe -c 1 -m 32 -g 1 \
  make mixture_mppde_train \
  | grep Log | cut -d ' ' -f3)

tails -l 2 $l1 $l2 $l3 $l4 $l5 $l6 $l7 $l8 $l9
