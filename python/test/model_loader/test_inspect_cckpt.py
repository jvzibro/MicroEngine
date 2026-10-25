from model_loader.inspect_ckpt import CheckpointInspector



def test_CheckpointInspector():
    ckpt_folder_path = "/models/Qwen3.5-4B"
    inspector = CheckpointInspector(ckpt_folder_path)
    is_valid = inspector.verify_safetensors_files()
    if is_valid:
        inspector.inspect_safetensors_files()

if __name__ == "__main__":
    test_CheckpointInspector()