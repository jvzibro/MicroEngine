import logging
from pathlib import Path
from safetensors import safe_open

logger = logging.getLogger(__name__)

DTYPE_SIZE = {
    "F64": 8, "F32": 4 ,"F16": 2,
    "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1
}


class CheckpointInspector:
    def __init__(self, ckpt_folder_path):
        logger.info(f"Initializing CheckpointInspector with folder: {ckpt_folder_path}")
        self.ckpt_folder_path = ckpt_folder_path
        self.ckpt_folder = Path(ckpt_folder_path)
        self.safetensors_files = sorted(self.ckpt_folder.glob("*.safetensors"))

    def count_safetensors_files(self):
        return len(self.safetensors_files)

    def verify_safetensors_files(self):
        self.num_safetensors_files = self.count_safetensors_files()
        if self.num_safetensors_files == 0:
            raise FileNotFoundError("No safetensors files found in the specified folder.")

        first_safetensors = self.safetensors_files[0]
        first_st_name = first_safetensors.stem
        official_count_of_safetensors = int(first_st_name.rsplit("-of-", 1)[1])

        if self.num_safetensors_files != official_count_of_safetensors:
            raise ValueError(
                f"Number of safetensors files ({self.num_safetensors_files}) does not match the official count ({official_count_of_safetensors})."
            )

        return True

    def inspect_safetensors_files(self):
        logger.info("Starting inspection of safetensors files.")
        num_tensors = 0
        total_size_in_bytes = 0
        dtype_class = set()
        for safetensors_file in self.safetensors_files:
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                
                for key in f.keys():
                    sl = f.get_slice(key)
                    shape = sl.get_shape()
                    dtype = sl.get_dtype()
                    num_elements = 1
                    for dim in shape:
                        num_elements *= dim
                    size_in_bytes = num_elements * DTYPE_SIZE[dtype]
                    num_tensors += 1
                    total_size_in_bytes += size_in_bytes
                    dtype_class.add(dtype)

        logger.info(f"Num shards: {self.num_safetensors_files}")
        logger.info(f"Total tensors: {num_tensors}")
        logger.info(f"Total size: {total_size_in_bytes} bytes")
        logger.info(f"Dtype classes: {dtype_class}")






