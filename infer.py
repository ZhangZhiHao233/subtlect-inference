import argparse
from Network import NAFNet
import torch
import platform
import pydicom
from tqdm import tqdm
import torch.nn.functional as F
from pydicom.uid import ExplicitVRLittleEndian
from Utils import *

###
def inference(args):
    """Performs model inference and post-processing on input image volume.

        This is the core function for inference and image enhancement, including:
        normalization, model prediction, optional residual mixing (noise/texture),
        and sharpening via convolution. It is intended to be called by the
        application-level code.

        Args:
            args: An object containing the following attributes:
                - input_list (np.ndarray): A 3D numpy array of shape (N, H, W) representing
                  input CT slices.

                - *logger (optional): Logger instance for logging information.

                - device: Torch device to perform inference on (e.g., 'cuda' or 'cpu').
                - model: A PyTorch model (e.g., NAFNet) used for inference.
                - enable_fp16 (bool): Whether to use float16 precision for inference.
                - noise_mix_ratio (float): Weight for noise residual blending.
                - texture_mix_ratio (float): Weight for texture residual blending.
                - sharpening_factor (float): Sharpening strength (>1 applies sharpening).
                - batch_size (int): Number of slices to process per batch.

        Returns:
            np.ndarray: A 3D numpy array of shape (N, H, W) representing the
            post-processed output slices.
        """

    logger = args.logger
    # Options for inference
    device = args.device
    model = args.model
    input_volume = args.input_list  # (N, H, W)

    enable_fp16 = args.enable_fp16
    noise_mix_ratio = args.noise_mix_ratio
    texture_mix_ratio = args.texture_mix_ratio
    sharpening_factor = args.sharpening_factor
    batch_size = args.batch_size
    #############

    # Validate input
    if not isinstance(input_volume, np.ndarray) or input_volume.ndim != 3:
        logger.error("Expected input_list to be a 3D numpy array with shape (N, H, W)")
        raise ValueError("Expected input_list to be a 3D numpy array with shape (N, H, W)")

    # Optional: convert model to half precision
    dtype = torch.float32
    if enable_fp16:
        model = model.half()
        dtype = torch.float16

    # Build sharpening kernel
    sharpening_kernel_tensor = None
    if sharpening_factor > 1:
        # sharpen_factor = max(sharpening_factor, 2)
        side_value = (1 - sharpening_factor) / 4.0
        sharpening_kernel = np.array([
            [0, side_value, 0],
            [side_value, sharpening_factor, side_value],
            [0, side_value, 0]
        ], dtype=np.float32)
        sharpening_kernel_tensor = torch.tensor(sharpening_kernel, dtype=dtype, device=device).unsqueeze(0).unsqueeze(0)

    # Prepare for batched inference: determine number of slices and batches, and initialize output list
    total_samples = input_volume.shape[0]
    num_batches = (total_samples + batch_size - 1) // batch_size
    outputs = []

    with tqdm(total=total_samples, desc="Inference", unit='slice') as pbar:
        for batch_idx in range(num_batches):

            # Determine start and end indices for the current batch
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)

            # Normalize and prepare the input batch as a NumPy array (B, H, W)
            input_batch_np = Normalize(input_volume[start_idx:end_idx])
            input_batch = torch.tensor(input_batch_np, dtype=dtype, device=device).unsqueeze(1)

            # Expand to three channels to match model input requirements (B, 3, H, W)
            input_batch_three_channel = input_batch.expand(-1, 3, -1, -1)

            with torch.no_grad():
                output_batch_three_channel = model(input_batch_three_channel)

            # Convert three-channel output back to single-channel by averaging (B, 1, H, W)
            output_batch = output_batch_three_channel.mean(dim=1, keepdim=True)

            # Optional noise input mixing
            if noise_mix_ratio > 0:
                output_batch = output_batch * (1 - noise_mix_ratio) + input_batch * noise_mix_ratio
                output_batch = torch.clamp(output_batch, 0, 1)

            # Optional sharpening
            if sharpening_kernel_tensor is not None:
                output_batch = F.pad(output_batch, (1, 1, 1, 1), mode="replicate")
                output_batch = F.conv2d(output_batch, sharpening_kernel_tensor)

                # Optional texture input mixing
                if texture_mix_ratio > 0:
                    output_batch = output_batch * (1 - texture_mix_ratio) + input_batch * texture_mix_ratio
                    output_batch = torch.clamp(output_batch, 0, 1)

            # Convert output tensor to NumPy array (B, H, W) and de-normalize intensity range
            output = output_batch[:, 0, :, :].cpu().numpy()
            output = np.int16(DeNormalize(output))
            outputs.append(output)
            pbar.update(end_idx - start_idx)

    # (N, H, W)
    return np.concatenate(outputs, axis=0)  # (N, H, W)


def run(args):
    logger = args.logger
    for series in args.series_list:

        rel_path = series['rel_path']
        index_in_folder = series['index']
        series_uid = series['series_uid']
        num_files = len(series['files'])

        print(rel_path)
        logger.info(f"📦 Processing series | Path: {rel_path} | Index: {index_in_folder} | UID: {series_uid} | Slices: {num_files}")
        rel_path_updated = os.path.join(*rel_path.split(os.path.sep)[1:])
        print(rel_path_updated)
        output_folder = os.path.join(args.Output_path, f"{rel_path_updated}_series_{index_in_folder}")
        print(output_folder)
        os.makedirs(output_folder, exist_ok=True)

        dcm_list = []
        input_list = []
        with tqdm(total=len(series['files']), desc="Reading") as pbar:
            for dcm_path in series['files']:
                dcm = pydicom.dcmread(dcm_path)
                dcm_list.append(dcm)
                ct = pixel2CT(dcm.pixel_array, dcm.RescaleSlope, dcm.RescaleIntercept)
                input_list.append(ct)
                pbar.update(1)

        input_list = np.array(input_list)
        args.input_list = input_list
        output_list = inference(args)

        uid = pydicom.uid.generate_uid()
        with tqdm(total=output_list.shape[0], desc="Saving") as pbar:
            for i in range(output_list.shape[0]):
                image = output_list[i]
                dcm = dcm_list[i]
                dcm.Rows, dcm.Columns = image.shape

                # Ensure TransferSyntaxUID is explicitly set and uncompressed for safe saving;
                # required when original file lacks it or uses a compressed format
                if not hasattr(dcm.file_meta, 'TransferSyntaxUID') or dcm.file_meta.TransferSyntaxUID.is_compressed:
                    dcm.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

                pixel_data = np.asarray(image - dcm.RescaleIntercept) / dcm.RescaleSlope
                dcm.PixelData = pixel_data.astype(dcm.pixel_array.dtype).tobytes()

                series_desc = getattr(dcm, 'SeriesDescription', '').strip()
                dcm.SeriesDescription = f'{series_desc}_SCT_mix1_{args.noise_mix_ratio}_sharpen_{args.sharpening_factor}_mix2_{args.texture_mix_ratio}'
                dcm.SeriesInstanceUID = uid

                save_path = os.path.join(output_folder, os.path.basename(series['files'][i]))
                dcm.save_as(save_path)
                pbar.update(1)


def load_model(args):
    logger = args.logger
    model_structure = \
        r"""
          img_channel: 3
          width: 64
          enc_blk_nums: [2, 2, 4, 8]
          middle_blk_num: 12
          dec_blk_nums: [2, 2, 2, 2]
        """

    try:
        x = yaml.safe_load(model_structure)
        NAFNetModel = NAFNet(**x)

        checkpoint = torch.load(args.weights_path, map_location=args.device, weights_only=True)
        NAFNetModel.load_state_dict(checkpoint['params'])
        NAFNetModel.eval()
        NAFNetModel.to(args.device)

        logger.info(f"✅ Model loaded successfully from: {args.weights_path}")
        logger.info(f"🧠 Model structure: {NAFNetModel.__class__.__name__} with config: {x}")
        logger.info(f"💻 Running on device: {args.device}")
        args.model = NAFNetModel

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        sys.exit(1)


def main(arg_parser):

    if platform.system() == "Darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logger(log_path=os.path.join("log", "inference.log"))
    logger.info("🚀 Starting SubtleCT Inference")

    args = arg_parser.parse_args()
    args.device = device
    args.logger = logger

    check_paths(args)
    check_config_yaml(args)
    load_model(args)
    read_dicom_series_recursive(args)
    run(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='SubtleCT_DNE')
    parser.add_argument('--weights_path', type=str, default='checkpoint/20241030-ct2.pth', help='Path to model weights')
    parser.add_argument('--Input_path', type=str, default='Input', help='Path to the input')
    parser.add_argument('--Output_path', type=str, default='Output', help='Path to the output')
    parser.add_argument('--Config_path', type=str, default='config.yaml', help='Path to the config')
    main(parser)