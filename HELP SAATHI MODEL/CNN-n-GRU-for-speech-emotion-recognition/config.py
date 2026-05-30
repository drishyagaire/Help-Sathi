#!/usr/bin/env python
# coding: utf-8

"""
Configuration settings for Speech Emotion Recognition models
"""

import os

# Dataset selection - change this to switch between datasets
# Options: "tess", "iemocap", "ravdess", or "nepali"
DATASET = "nepali"

# Directory paths - update these to your local paths
# Recommended structure is to place datasets in a 'datasets' folder
# Example structure:
# - project-root/
#   - datasets/
#     - TESS/
#     - IEMOCAP/
#     - RAVDESS/
#   - models/
#   - utils/
#   - ...

# Use environment variables if available, otherwise use default relative paths
TESS_DATA_FOLDER = os.environ.get("TESS_DATA_PATH", "../datasets/TESS")
IEMOCAP_DATA_FOLDER = os.environ.get("IEMOCAP_DATA_PATH", "../datasets/IEMOCAP")
RAVDESS_DATA_FOLDER = os.environ.get("RAVDESS_DATA_PATH", "../datasets/RAVDESS")
NEPALI_DATA_FOLDER = os.environ.get("NEPALI_DATA_PATH", "dataset/nepali_dataset_preprocessed")
NEPALI_DATA_CSV = os.environ.get("NEPALI_DATA_CSV", "dataset/nepali_dataset_preprocessed.csv")

# Model hyperparameters
BATCH_SIZE = 32
HIDDEN_DIM = 64
GRU_LAYERS = 1
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Experiment settings
EXPERIMENTS_FOLDER = "experiments"

# TESS dataset settings
TESS_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
TESS_EMOTION_TO_IDX = {emotion: i for i, emotion in enumerate(TESS_EMOTIONS)}
TESS_IDX_TO_EMOTION = {i: emotion for i, emotion in enumerate(TESS_EMOTIONS)}

# IEMOCAP dataset settings
IEMOCAP_EMOTIONS = ['ang', 'hap', 'neu', 'sad']
IEMOCAP_EMOTION_TO_IDX = {emotion: i for i, emotion in enumerate(IEMOCAP_EMOTIONS)}
IEMOCAP_IDX_TO_EMOTION = {i: emotion for i, emotion in enumerate(IEMOCAP_EMOTIONS)}

# RAVDESS dataset settings
RAVDESS_EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
RAVDESS_EMOTION_TO_IDX = {emotion: i for i, emotion in enumerate(RAVDESS_EMOTIONS)}
RAVDESS_IDX_TO_EMOTION = {i: emotion for i, emotion in enumerate(RAVDESS_EMOTIONS)} 

# Nepali dataset settings
NEPALI_EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NEPALI_EMOTION_TO_IDX = {emotion: i for i, emotion in enumerate(NEPALI_EMOTIONS)}
NEPALI_IDX_TO_EMOTION = {i: emotion for i, emotion in enumerate(NEPALI_EMOTIONS)}

# Nepali audio cleanup settings
NEPALI_SAMPLE_RATE = 16000
NEPALI_MAX_RAW_DURATION_SECONDS = 7.0
NEPALI_MIN_CLEAN_DURATION_SECONDS = 0.35
NEPALI_TRIM_TOP_DB = 25
