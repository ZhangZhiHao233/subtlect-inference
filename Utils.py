import logging
import os
import sys
import yaml
import SimpleITK as sitk
import numpy as np

def setup_logger(log_path='subtlect_inf.log', log_level=logging.INFO):
    logger = logging.getLogger("SubtleCT")
    logger.setLevel(log_level)
    logger.propagate = False  # Prevent duplicate printing

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)

    # Log formatting
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers (avoid duplicate additions)
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


def check_paths(args):
    logger = args.logger
    # Check model weights
    if not os.path.isfile(args.weights_path):
        logger.error(f"❌ [Weights] File not found: {args.weights_path}")
        sys.exit(1)
    logger.info(f"✅ [Weights] Loaded from: {args.weights_path}")

    # Check input directory
    if not os.path.isdir(args.Input_path):
        logger.error(f"❌ [Input] Directory not found: {args.Input_path}")
        sys.exit(1)
    logger.info(f"✅ [Input] Found input directory: {args.Input_path}")

    # Check or create output directory
    if not os.path.exists(args.Output_path):
        try:
            os.makedirs(args.Output_path, exist_ok=True)
            logger.info(f"✅ [Output] Created output directory: {args.Output_path}")
        except Exception as e:
            logger.error(f"❌ [Output] Failed to create output directory: {args.Output_path} — {e}")
            sys.exit(1)
    else:
        logger.info(f"✅ [Output] Using existing output directory: {args.Output_path}")

    # Check config file
    if not os.path.isfile(args.Config_path):
        logger.error(f"❌ [Config] Configuration file not found: {args.Config_path}")
        sys.exit(1)
    logger.info(f"✅ [Config] Loaded configuration file: {args.Config_path}")


def check_config_yaml(args):
    logger = args.logger
    if not os.path.isfile(args.Config_path):
        raise FileNotFoundError(f"❌ Config file not found: {args.Config_path}")

    with open(args.Config_path, 'r') as f:
        config = yaml.safe_load(f)

    required_keys = {
        "version": str,
        "noise_mix_ratio": (int, float),
        "sharpening_factor": (int, float),
        "texture_mix_ratio": (int, float),
        "batch_size": int,
        "enable_fp16": bool
    }

    for key, expected_type in required_keys.items():
        if key not in config:
            raise KeyError(f"❌ Missing required config key: '{key}'")
        if not isinstance(config[key], expected_type):
            # Format multiple expected types
            expected_name = (
                "/".join([t.__name__ for t in expected_type])
                if isinstance(expected_type, tuple) else expected_type.__name__
            )
            raise TypeError(
                f"❌ Key '{key}' should be of type {expected_name}, "
                f"but got {type(config[key]).__name__}"
            )

    for key, value in config.items():
        setattr(args, key, value)

    logger.info("✅ YAML config check passed and values loaded into args.")
    return config


def read_dicom_series_recursive(args, min_slices=1):

    logger = args.logger
    """
    Recursively scan input_root to extract all DICOM series.
    Returns: List[Dict], each dict includes 'rel_path', 'index', 'series_uid', 'files'
    """
    series_list = []

    for root, dirs, files in os.walk(args.Input_path):
        if not files:
            continue

        reader = sitk.ImageSeriesReader()
        try:
            series_ids = reader.GetGDCMSeriesIDs(root)
        except Exception as e:
            logger.warning(f"⚠️ Failed to read series under {root}: {e}")
            continue

        if not series_ids:
            continue

        rel_path = root
        logger.info(f"📂 Found {len(series_ids)} series in {rel_path}")

        for i, series_uid in enumerate(series_ids):
            series_file_names = reader.GetGDCMSeriesFileNames(root, series_uid)
            if len(series_file_names) < min_slices:
                logger.warning(f"  ⚠️ Skipped UID={series_uid}, not enough slices: {len(series_file_names)}")
                continue

            try:
                reader.SetFileNames(series_file_names)
                _ = reader.Execute()  # Validate series can be read
            except Exception as e:
                logger.error(f"  ❌ Failed to read UID={series_uid}: {e}")
                continue

            series_list.append({
                "rel_path": rel_path,
                "index": i + 1,
                "series_uid": series_uid,
                "files": series_file_names
            })

            logger.info(f"  ✅ Series {i+1}: UID={series_uid} | Slices: {len(series_file_names)}")

    args.series_list = series_list


def pixel2CT(pixel_array, RescaleSlope, RescaleIntercept, CT_Min=-1024, CT_Max=3071):
    CT_matrix = pixel_array * RescaleSlope + RescaleIntercept
    CT_matrix = np.clip(CT_matrix, CT_Min, CT_Max)
    return CT_matrix

def Normalize(CT_matrix, CT_Min=-1024, CT_Max=3071):
    CT_matrix_Norm = (CT_matrix - CT_Min) / (CT_Max - CT_Min)
    return CT_matrix_Norm

def DeNormalize(CT_matrix_Norm, CT_Min=-1024, CT_Max=3071):
    CT_matrix_Norm = np.clip(CT_matrix_Norm, 0, 1)
    CT_matrix = CT_matrix_Norm * (CT_Max - CT_Min) + CT_Min
    return CT_matrix

def CT2pixel(CT_matrix, RescaleSlope, RescaleIntercept):
    pixel_array = (CT_matrix - RescaleIntercept) / RescaleSlope
    return pixel_array