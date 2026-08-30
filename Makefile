SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PROFILE ?= smoke
DEVICE ?= cpu
WORLD_SIZE ?= 1
LOCAL_WORLD_SIZE ?= $(WORLD_SIZE)
NNODES ?= 1
NODE_RANK ?= 0
MASTER_ADDR ?= 127.0.0.1
MASTER_PORT ?= 29500
DATA_ROOT ?= $(CURDIR)/data/memx
MODEL_ROOT ?= $(CURDIR)/.cache/memx/models
RUN_ROOT ?= $(CURDIR)/artifacts/company/$(PROFILE)
CONFIG ?= configs/experiments/$(PROFILE).yaml
DATA_CONFIG ?= configs/data/$(PROFILE).yaml
RESUME ?= never
OFFLINE ?= 0

.PHONY: help bootstrap data models smoke train evaluate report

help:
	@echo "memX reproducible execution targets"
	@echo "  make bootstrap  Validate/install the selected runtime"
	@echo "  make data       Prepare and hash the selected dataset"
	@echo "  make models     Download and verify pinned SANA/DINO snapshots"
	@echo "  make smoke      Run data, train, evaluate, and report offline"
	@echo "  make train      Launch torchrun training"
	@echo "  make evaluate   Launch torchrun evaluation"
	@echo "  make report     Render validated JSON, CSV, and Markdown"
	@echo "Variables: PROFILE DEVICE WORLD_SIZE LOCAL_WORLD_SIZE NNODES NODE_RANK"
	@echo "           MASTER_ADDR MASTER_PORT DATA_ROOT MODEL_ROOT RUN_ROOT CONFIG DATA_CONFIG"
	@echo "           RESUME OFFLINE"

bootstrap:
	DEVICE="$(DEVICE)" scripts/bootstrap.sh

data:
	@if [[ "$(OFFLINE)" == "1" ]]; then \
		scripts/run_memx.sh data prepare --config "$(DATA_CONFIG)" --root "$(DATA_ROOT)" --offline; \
	elif [[ "$(OFFLINE)" == "0" ]]; then \
		scripts/run_memx.sh data prepare --config "$(DATA_CONFIG)" --root "$(DATA_ROOT)"; \
	else \
		echo "OFFLINE must be 0 or 1" >&2; exit 2; \
	fi

models:
	scripts/run_memx.sh model prepare --config configs/pilot/sana-1.5-1.6b.json --root "$(MODEL_ROOT)"

smoke:
	scripts/run_memx.sh smoke --config "$(CONFIG)" --data-root "$(DATA_ROOT)" --run-root "$(RUN_ROOT)" --device "$(DEVICE)"

train:
	MEMX_MODE=train DEVICE="$(DEVICE)" WORLD_SIZE="$(WORLD_SIZE)" LOCAL_WORLD_SIZE="$(LOCAL_WORLD_SIZE)" NNODES="$(NNODES)" NODE_RANK="$(NODE_RANK)" MASTER_ADDR="$(MASTER_ADDR)" MASTER_PORT="$(MASTER_PORT)" DATA_ROOT="$(DATA_ROOT)" MODEL_ROOT="$(MODEL_ROOT)" RUN_ROOT="$(RUN_ROOT)" CONFIG="$(CONFIG)" RESUME="$(RESUME)" scripts/launch_train.sh

evaluate:
	MEMX_MODE=evaluate DEVICE="$(DEVICE)" WORLD_SIZE="$(WORLD_SIZE)" LOCAL_WORLD_SIZE="$(LOCAL_WORLD_SIZE)" NNODES="$(NNODES)" NODE_RANK="$(NODE_RANK)" MASTER_ADDR="$(MASTER_ADDR)" MASTER_PORT="$(MASTER_PORT)" DATA_ROOT="$(DATA_ROOT)" MODEL_ROOT="$(MODEL_ROOT)" RUN_ROOT="$(RUN_ROOT)" CONFIG="$(CONFIG)" scripts/launch_train.sh

report:
	scripts/run_memx.sh report --run-root "$(RUN_ROOT)"
