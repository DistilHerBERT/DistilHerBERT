#!/bin/bash
srun --time 3-0 --gres=gpu python pl_trainer.py
