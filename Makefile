
###################
# General setting #
###################

# Global settings
CUDA = cu111  # CUDA version 11.1
PYTHON ?= poetry run python3
RUN = poetry run
PIP = pip3
RM = rm -rf

# User settings
DEVICE_ID = 0


#################
# Model setting #
#################

# # Setting for FluxGNN
CD_FLUXGNN_INPUT_YAML ?= inputs/convection_diffusion/fluxgnn/fluxgnn.yml
CD_FLUXGNN_PRETRAINED_MODEL ?= data/pretrained/convection_diffusion/fluxgnn

MIXTURE_FLUXGNN_STEP ?= 4  # 2, 4
MIXTURE_FLUXGNN_NODE ?= 8  # 8, 16, 32, 64
MIXTURE_FLUXGNN_REP ?= 4  # 2, 4, 8
MIXTURE_FLUXGNN_PRETRAINED_MODEL ?= data/pretrained/mixture/fluxgnn_step4_n8_rep4


# # Setting for FVM
CD_FV_INPUT_YAML ?= inputs/convection_diffusion/fv/fv.yml

MIXTURE_FVM_UREP ?= 32  # 4, 8, 16, 32, 64, 128
MIXTURE_FVM_AREP ?= 4  # 4, 8, 16, 32, 64, 128


# # Setting for PENN
PENN_N_NODE ?= 32  # 4 8 16 32 64 128
PENN_REP ?= 4  # 4 8 16
MIXTURE_PENN_PRETRAINED_MODEL ?= data/pretrained/mixture/penn_n32_rep4


# # Setting for MP-PDE
MPPDE_TW ?= 4  # 2 4 8
MPPDE_HIDDEN_FEATURES ?= 128  # 32 64 128
MPPDE_NEIGHBORS ?= 4  # 4 8 16
MIXTURE_MPPDE_PRETRAINED_MODEL ?= data/pretrained/mixture/mppde_f128_n4_tw4


################
# Installation #
################

install: libs
	-$(PYTHON) -m pip uninstall -y mp-pde-solvers
	make -C mp-neural-pde-solvers install

libs: poetry
	-$(PYTHON) -m pip uninstall -y pysiml
	$(PYTHON) -m pip install lib/siml/dist/pysiml-*.whl
	$(PYTHON) -m pip install pytest

poetry:
	cd lib/siml && poetry build --format=wheel && cd ../..
	poetry config virtualenvs.create true
	poetry config virtualenvs.in-project true
	-$(PYTHON) -m pip uninstall -y torch torch-scatter torch-sparse torch-geometric
	-$(PYTHON) -m pip uninstall -y fluxgnn
	poetry install
	poetry run python3 -m pip install torch==1.9.1+cu111 --extra-index-url https://download.pytorch.org/whl/cu111


################################
# Convection-diffusion dataset #
################################

cd_fluxgnn_train:
	$(PYTHON) experiments/run.py train \
		$(CD_FLUXGNN_INPUT_YAML)

cd_fluxgnn_eval:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/run.py eval \
		$(CD_FLUXGNN_PRETRAINED_MODEL) -g -1

cd_fvm_eval:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/run.py fv \
		$(CD_FV_INPUT_YAML) -g -1


###################
# Mixture dataset #
###################

## Ours
mixture_fluxgnn_train:
	$(PYTHON) experiments/run.py train \
		inputs/mixture/fluxgnn/fluxgnn_step$(MIXTURE_FLUXGNN_STEP)_n$(MIXTURE_FLUXGNN_NODE)_rep$(MIXTURE_FLUXGNN_REP).yml

mixture_fluxgnn_eval: \
	mixture_fluxgnn_eval_ref \
	mixture_fluxgnn_eval_rotation \
	mixture_fluxgnn_eval_scaling \
	mixture_fluxgnn_eval_taller

mixture_fluxgnn_eval_ref:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/run.py eval \
		$(MIXTURE_FLUXGNN_PRETRAINED_MODEL) -g -1

mixture_fluxgnn_eval_rotation:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/run.py eval \
		$(MIXTURE_FLUXGNN_PRETRAINED_MODEL) -g -1 \
	 -d data/mixture/transformed/raw/test/rotation/h* \
	 -n rotation

mixture_fluxgnn_eval_scaling:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/run.py eval \
		$(MIXTURE_FLUXGNN_PRETRAINED_MODEL) -g -1 \
	 -d data/mixture/transformed/raw/test/scaling/h* \
	 -n scaling

mixture_fluxgnn_eval_taller:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/run.py eval \
		$(MIXTURE_FLUXGNN_PRETRAINED_MODEL) -g -1 \
	 -d data/mixture/taller/raw/h* \
	 -n taller


## FVM
mixture_fvm_eval: \
	mixture_fvm_eval_ref \
	mixture_fvm_eval_rotation \
	mixture_fvm_eval_scaling \
	mixture_fvm_eval_taller

mixture_fvm_eval_ref:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/run.py fv \
		inputs/mixture/fv/fv_urep$(MIXTURE_FVM_UREP)_arep$(MIXTURE_FVM_AREP).yml \
		-g -1

mixture_fvm_eval_rotation:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/run.py fv \
		inputs/mixture/fv/fv_urep$(MIXTURE_FVM_UREP)_arep$(MIXTURE_FVM_AREP).yml \
		-g -1 \
		-d data/mixture/transformed/raw/test/rotation/h* \
		-n rotation

mixture_fvm_eval_scaling:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/run.py fv \
		inputs/mixture/fv/fv_urep$(MIXTURE_FVM_UREP)_arep$(MIXTURE_FVM_AREP).yml \
		-g -1 \
		-d data/mixture/transformed/raw/test/scaling/h* \
		-n scaling

mixture_fvm_eval_taller:
	OMP_NUM_THREADS=1 $(PYTHON) experiments/run.py fv \
		inputs/mixture/fv/fv_urep$(MIXTURE_FVM_UREP)_arep$(MIXTURE_FVM_AREP).yml \
		-g -1 \
		-d data/mixture/taller/raw/h* \
		-n taller


## PENN (Horie+ NeurIPS 2022)
mixture_penn_train:
	$(PYTHON) penn/train.py \
		inputs/mixture/penn/penn_n$(strip $(PENN_N_NODE))_rep$(strip $(PENN_REP)).yml \
		-g $(DEVICE_ID) -c true \
		-o results/mixture/penn/models/penn_n$(strip $(PENN_N_NODE))_rep$(strip $(PENN_REP))

mixture_penn_eval: \
	mixture_penn_eval_ref \
	mixture_penn_eval_rotation \
	mixture_penn_eval_scaling \
	mixture_penn_eval_taller

mixture_penn_eval_ref:
	OMP_NUM_THREADS=1 $(PYTHON) penn/eval.py \
		$(MIXTURE_PENN_PRETRAINED_MODEL) \
		data/mixture/preprocessed/test \
		-b results/mixture/penn/predictions \
		-p data/mixture/preprocessed/preprocessors.pkl \
		-w data/mixture/interim \
		-a mixture \
		-e 32

mixture_penn_eval_rotation:
	OMP_NUM_THREADS=1 $(PYTHON) penn/eval.py \
		$(MIXTURE_PENN_PRETRAINED_MODEL) \
		data/mixture/transformed/preprocessed/test/rotation \
		-b results/mixture/penn/predictions \
		-p data/mixture/preprocessed/preprocessors.pkl \
		-w data/mixture/transformed/interim \
		-a mixture \
		-e 32

mixture_penn_eval_scaling:
	OMP_NUM_THREADS=1 $(PYTHON) penn/eval.py \
		$(MIXTURE_PENN_PRETRAINED_MODEL) \
		data/mixture/transformed/preprocessed/test/scaling \
		-b results/mixture/penn/predictions \
		-p data/mixture/preprocessed/preprocessors.pkl \
		-w data/mixture/transformed/interim \
		-a mixture \
		-e 32

mixture_penn_eval_taller:
	OMP_NUM_THREADS=1 $(PYTHON) penn/eval.py \
		$(MIXTURE_PENN_PRETRAINED_MODEL) \
		data/mixture/taller/preprocessed \
		-b results/mixture/penn/predictions \
		-p data/mixture/preprocessed/preprocessors.pkl \
		-w data/mixture/taller/interim \
		-a mixture \
		-e 32


## MP-PDE (Brandstetter+ ICLR 2022)
mixture_mppde_train:
	make -C mp-neural-pde-solvers train_tw$(strip $(MPPDE_TW)) \
		NEIGHBORS=$(strip $(MPPDE_NEIGHBORS)) \
		HIDDEN_FEATURES=$(strip $(MPPDE_HIDDEN_FEATURES))

mixture_mppde_eval: \
	mixture_mppde_eval_ref \
	mixture_mppde_eval_rotation \
	mixture_mppde_eval_scaling \
	mixture_mppde_eval_taller

mixture_mppde_eval_ref:
	OMP_NUM_THREADS=1 make -C mp-neural-pde-solvers eval_tw$(strip $(MPPDE_TW)) \
		SAVE_DIRECTORY=../results/mixture/mppde/$(notdir $(MIXTURE_MPPDE_PRETRAINED_MODEL)) \
		MODEL_PATH=../$(MIXTURE_MPPDE_PRETRAINED_MODEL)/model.pt \
		NEIGHBORS=$(strip $(MPPDE_NEIGHBORS)) \
		HIDDEN_FEATURES=$(strip $(MPPDE_HIDDEN_FEATURES))
	$(PYTHON) experiments/generate_vtu.py \
		results/mixture/mppde/$(notdir $(MIXTURE_MPPDE_PRETRAINED_MODEL)) \
		data/mixture/interim \
		-p data/mixture/preprocessed/preprocessors.pkl \
		-f true

mixture_mppde_eval_rotation:
	OMP_NUM_THREADS=1 make -C mp-neural-pde-solvers eval_tw$(MPPDE_TW) \
		SAVE_DIRECTORY=../results/mixture/mppde/rotation/$(notdir $(MIXTURE_MPPDE_PRETRAINED_MODEL)) \
		MODEL_PATH=../$(MIXTURE_MPPDE_PRETRAINED_MODEL)/model.pt \
		NEIGHBORS=$(strip $(MPPDE_NEIGHBORS)) \
		HIDDEN_FEATURES=$(MPPDE_HIDDEN_FEATURES) \
		DATA_TYPE=rotation
	$(PYTHON) experiments/generate_vtu.py \
		results/mixture/mppde/rotation/$(notdir $(MIXTURE_MPPDE_PRETRAINED_MODEL)) \
		data/mixture/transformed/interim/test \
		-p data/mixture/preprocessed/preprocessors.pkl \
		-f true

mixture_mppde_eval_scaling:
	OMP_NUM_THREADS=1 make -C mp-neural-pde-solvers eval_tw$(MPPDE_TW) \
		SAVE_DIRECTORY=../results/mixture/mppde/scaling/$(notdir $(MIXTURE_MPPDE_PRETRAINED_MODEL)) \
		MODEL_PATH=../$(MIXTURE_MPPDE_PRETRAINED_MODEL)/model.pt \
		NEIGHBORS=$(strip $(MPPDE_NEIGHBORS)) \
		HIDDEN_FEATURES=$(MPPDE_HIDDEN_FEATURES) \
		DATA_TYPE=scaling
	$(PYTHON) experiments/generate_vtu.py \
		results/mixture/mppde/scaling/$(notdir $(MIXTURE_MPPDE_PRETRAINED_MODEL)) \
		data/mixture/transformed/interim/test \
		-p data/mixture/preprocessed/preprocessors.pkl \
		-f true

mixture_mppde_eval_taller:
	OMP_NUM_THREADS=1 make -C mp-neural-pde-solvers eval_tw$(MPPDE_TW) \
		SAVE_DIRECTORY=../results/mixture/mppde/taller/$(notdir $(MIXTURE_MPPDE_PRETRAINED_MODEL)) \
		MODEL_PATH=../$(MIXTURE_MPPDE_PRETRAINED_MODEL)/model.pt \
		NEIGHBORS=$(strip $(MPPDE_NEIGHBORS)) \
		HIDDEN_FEATURES=$(MPPDE_HIDDEN_FEATURES) \
		DATA_TYPE=taller
	$(PYTHON) experiments/generate_vtu.py \
		results/mixture/mppde/taller/$(notdir $(MIXTURE_MPPDE_PRETRAINED_MODEL)) \
		data/mixture/taller/interim \
		-p data/mixture/preprocessed/preprocessors.pkl \
		-f true


########
# Test #
########
test_cd_fluxgnn_train:
	$(PYTHON) experiments/run.py train \
		tests/cd/fluxgnn.yml -g -1

test_mixture_fluxgnn_train:
	$(PYTHON) experiments/run.py train \
		tests/mixture/fluxgnn.yml -g -1


########
# Misc #
########

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Delete all data
delete_all_data: clean
	$(RM) data/convection_diffusion/data/*
	$(RM) data/mixture/data/*
