import os
import subprocess
from typing import Iterable, Tuple, List, Tuple

import torch
from datasets.imagenet_util import (
    evaluate_ppq_module_with_imagenet,
    load_imagenet_from_directory,
)
from esp_ppq import QuantizationSettingFactory, QuantizationSetting
from esp_ppq.api import espdl_quantize_onnx, get_target_platform
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
import urllib.request
import zipfile


def quant_setting_mobilenet_v2(
    onnx_path: str,
    optim_quant_method: List[str] = None,
) -> Tuple[QuantizationSetting, str]:
    """
    Modified to support default settings by returning standard espdl_setting
    when optim_quant_method is None.
    """
    quant_setting = QuantizationSettingFactory.espdl_setting()
    
    # If None is passed, we skip all logic below and return the default config
    if optim_quant_method is None:
        return quant_setting, onnx_path

    if "MixedPrecision_quantization" in optim_quant_method:
        quant_setting.dispatching_table.append(
            "/features/features.1/conv/conv.0/conv.0.0/Conv",
            get_target_platform(TARGET, 16),
        )
        quant_setting.dispatching_table.append(
            "/features/features.1/conv/conv.0/conv.0.2/Clip",
            get_target_platform(TARGET, 16),
        )
    elif "LayerwiseEqualization_quantization" in optim_quant_method:
        quant_setting.equalization = True
        quant_setting.equalization_setting.iterations = 6
        quant_setting.equalization_setting.value_threshold = 0.5
        quant_setting.equalization_setting.opt_level = 2
        quant_setting.equalization_setting.interested_layers = None
        # Note: LEQ usually requires the ReLU version of the model
        onnx_path = onnx_path.replace("mobilenet_v2.onnx", "mobilenet_v2_relu.onnx")
    
    return quant_setting, onnx_path


def collate_fn1(x: Tuple) -> torch.Tensor:
    return torch.cat([sample[0].unsqueeze(0) for sample in x], dim=0)


def collate_fn2(batch: torch.Tensor) -> torch.Tensor:
    return batch.to(DEVICE)


def report_hook(blocknum, blocksize, total):
    downloaded = blocknum * blocksize
    percent = downloaded / total * 100
    print(f"\rDownloading calibration dataset: {percent:.2f}%", end="")


if __name__ == "__main__":
    BATCH_SIZE = 32
    INPUT_SHAPE = [3, 224, 224]
    DEVICE = "cpu"  #  'cuda' or 'cpu', if you use cuda, please make sure that cuda is available
    TARGET = "esp32s3"  #  'c', 'esp32s3' or 'esp32p4'
    NUM_OF_BITS = 16
    ONNX_PATH = "./models/mobilenet_v2.onnx"  #'models/onnx/mobilenet_v2.onnx'
    ESPDL_MODEL_PATH = "./models/mobilenet_v2_16bit.espdl"
    CALIB_DIR = "./imagenet"

    # Download mobilenet_v2 model from onnx models and dataset
    imagenet_url = "https://dl.espressif.com/public/imagenet_calib.zip"
    os.makedirs(CALIB_DIR, exist_ok=True)
    if not os.path.exists("imagenet_calib.zip"):
        urllib.request.urlretrieve(
            imagenet_url, "imagenet_calib.zip", reporthook=report_hook
        )
    if not os.path.exists(os.path.join(CALIB_DIR, "calib")):
        with zipfile.ZipFile("imagenet_calib.zip", "r") as zip_file:
            zip_file.extractall(CALIB_DIR)
    CALIB_DIR = os.path.join(CALIB_DIR, "calib")

    # -------------------------------------------
    # Prepare Calibration Dataset
    # --------------------------------------------
    if os.path.exists(CALIB_DIR):
        print(f"load imagenet calibration dataset from directory: {CALIB_DIR}")
        dataset = datasets.ImageFolder(
            CALIB_DIR,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        dataset = Subset(dataset, indices=[_ for _ in range(0, 1024)])
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            collate_fn=collate_fn1,
        )
    else:
        # Random calibration dataset only for debug
        print("load random calibration dataset")

        def load_random_calibration_dataset() -> Iterable:
            return [torch.rand(size=INPUT_SHAPE) for _ in range(BATCH_SIZE)]

        # Load training data for creating a calibration dataloader.
        dataloader = DataLoader(
            dataset=load_random_calibration_dataset(),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

    # -------------------------------------------
    # Quantize ONNX Model.
    # --------------------------------------------

    # create a setting for quantizing your network with ESPDL.
    # if you don't need to optimize quantization, set the input 1 of the quant_setting_mobilenet_v2 function None
    # Example: Using LayerwiseEqualization_quantization
    quant_setting, ONNX_PATH = quant_setting_mobilenet_v2(
        ONNX_PATH, None
    )

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_PATH,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=[1] + INPUT_SHAPE,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn2,
        setting=quant_setting,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=1,
    )
    

    # -------------------------------------------
    # Evaluate Quantized Model.
    # --------------------------------------------

    evaluate_ppq_module_with_imagenet(
        model=quant_ppq_graph,
        imagenet_validation_dir=CALIB_DIR,
        batchsize=BATCH_SIZE,
        device=DEVICE,
        verbose=1,
    )
