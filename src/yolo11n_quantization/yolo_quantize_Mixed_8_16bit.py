import os
from esp_ppq import QuantizationSettingFactory
from esp_ppq.api import espdl_quantize_onnx, get_target_platform
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from onnxsim import simplify
import onnx
import zipfile
import urllib.request


class CaliDataset(Dataset):
    def __init__(self, path, img_shape=640):
        super().__init__()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((img_shape, img_shape)),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ]
        )

        self.imgs_path = []
        self.path = path
        for img_name in os.listdir(self.path):
            img_path = os.path.join(self.path, img_name)
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img = Image.open(self.imgs_path[idx])
        img = self.transform(img)
        return img


def report_hook(blocknum, blocksize, total):
    downloaded = blocknum * blocksize
    percent = downloaded / total * 100
    print(f"\rDownloading calibration dataset: {percent:.2f}%", end="")


def quant_yolo11n(imgsz):
    BATCH_SIZE = 32
    INPUT_SHAPE = [3, imgsz, imgsz]
    DEVICE = "cpu"
    TARGET = "esp32s3"
    NUM_OF_BITS = 8
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ONNX_PATH = os.path.join(
        script_dir, "./models/yolo11n.onnx"
    )
    ESPDL_MODLE_PATH = os.path.join(
        script_dir,
        "./models/yolo11n_s8_mixed_precision.espdl",  # ← separate output file
    )

    yolo11n_caib_url = "https://dl.espressif.com/public/calib_yolo11n.zip"
    CALIB_DIR = "calib_yolo11n"
    if not os.path.exists(CALIB_DIR):
        urllib.request.urlretrieve(
            yolo11n_caib_url, "calib_yolo11n.zip", reporthook=report_hook
        )
        print()
        with zipfile.ZipFile("calib_yolo11n.zip", "r") as zip_file:
            zip_file.extractall("./")

    model = onnx.load(ONNX_PATH)
    sim = True
    if sim:
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
    onnx.save(onnx.shape_inference.infer_shapes(model), ONNX_PATH)

    calibration_dataset = CaliDataset(CALIB_DIR, img_shape=imgsz)
    dataloader = DataLoader(
        dataset=calibration_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        return batch.to(DEVICE)

    # Mixed-Precision Quantization
    # Layers elevated to INT16 based on graphwise noise-to-signal analysis:
    # C2PSA attention module (model.10): FFN and attention projection layers
    # Detection head (model.23): cv2 and cv3 branch layers exceeding 20% graphwise error
    quant_setting = QuantizationSettingFactory.espdl_setting()

    high_error_layers_16bit = [
        # C2PSA attention module — highest graphwise errors (36.0%, 28.7%)
        '/model.10/m/m.0/ffn/ffn.1/conv/Conv',
        '/model.10/m/m.0/attn/proj/conv/Conv',
        '/model.10/m/m.0/attn/qkv/conv/Conv',
        '/model.10/m/m.0/attn/pe/conv/Conv',
        '/model.10/m/m.0/attn/MatMul_1',
        '/model.10/m/m.0/attn/MatMul',
        '/model.10/m/m.0/ffn/ffn.0/conv/Conv',
        '/model.10/cv1/conv/Conv',
        '/model.10/cv2/conv/Conv',
        # Detection head model.23 — cv2 and cv3 branches >20% graphwise error
        '/model.23/cv3.2/cv3.2.0/cv3.2.0.0/conv/Conv',
        '/model.23/cv2.2/cv2.2.0/conv/Conv',
        '/model.23/cv3.2/cv3.2.1/cv3.2.1.1/conv/Conv',
        '/model.23/cv3.2/cv3.2.0/cv3.2.0.1/conv/Conv',
        '/model.23/cv2.2/cv2.2.1/conv/Conv',
        '/model.23/cv3.0/cv3.0.1/cv3.0.1.1/conv/Conv',
        '/model.23/cv3.1/cv3.1.1/cv3.1.1.1/conv/Conv',
        '/model.23/cv2.1/cv2.1.1/conv/Conv',
    ]

    for layer in high_error_layers_16bit:
        quant_setting.dispatching_table.append(
            operation=layer,
            platform=get_target_platform(TARGET, 16)
        )

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_PATH,
        espdl_export_file=ESPDL_MODLE_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,
        input_shape=[1] + INPUT_SHAPE,
        target=TARGET,
        num_of_bits=NUM_OF_BITS,
        collate_fn=collate_fn,
        setting=quant_setting,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=False,
        verbose=0,
        inputs=None,
    )
    return quant_ppq_graph


if __name__ == "__main__":
    quant_yolo11n(imgsz=640)