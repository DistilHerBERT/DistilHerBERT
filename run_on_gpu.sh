#!/bin/bash
srun --time 2-0 --gres=gpu --gpus 1 python pl_trainer.py
