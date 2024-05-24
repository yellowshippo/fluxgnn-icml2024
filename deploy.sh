#!/bin/bash
set -eu

# Clone
mkdir -p deploy
rm -rf deploy/fluxgnn_icml2024
cd deploy
git clone git@gitlab.ritc.jp:ricos/machine_learning/dggnn.git -b icml2024 fluxgnn_icml2024 --recurse-submodules
cd fluxgnn_icml2024

# Remove unnecessary files
rm -rf \
  .git* \
  *.sh \
  lib/siml/.git* \
  lib/siml/LICENSE \
  lib/siml/README.md \
  lib/siml/docker \
  lib/siml/docs \
  lib/siml/examples \
  lib/siml/sphinx \
  lib/siml/tests \
  lib/femio/.git* \
  lib/femio/LICENSE \
  lib/femio/README.md \
  lib/femio/docker \
  lib/femio/docs \
  lib/femio/examples \
  lib/femio/sphinx \
  lib/femio/tests \
  data/*

cat lib/siml/pyproject.toml | grep -v ricos | sed 's/"RICOS Co. Ltd."//g' > tmp
mv tmp lib/siml/pyproject.toml
touch lib/siml/README.md

cat lib/femio/pyproject.toml | grep -v ricos | sed 's/"RICOS Co. Ltd."//g' > tmp
mv tmp lib/femio/pyproject.toml
touch lib/femio/README.md

touch lib/siml/.git lib/femio/.git .git

grep -i ricos $(find . -type f) || true
grep -i horie $(find . -type f)
