from models.convnext import ConvNeXt, ConvNeXtConfig
from models.transformer import GPTConfig
from classifiers.classifier import ShotPredictor
from loaders.dataloader import DataLoaderStage0, DataLoaderStage1, DataLoaderStage2
import torch
import numpy as np
import pandas as pd
import argparse
from loaders.vocab import ITOSSTAGE1, ITOSSTAGE2

STAGE2 = True


def eval(stage, model=None):
    """stage: "stage1", "stage2" """
    assert (
        stage == "stage2"
    ), "this is a continuation from CNN+RNN approach, we assumed stage1 ConvNeXt is already pre-trained"

    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        # conv  = ResNet_RS()
        conv_config = ConvNeXtConfig()
        config = GPTConfig()
        model = ShotPredictor(config_conv=conv_config, config_transformer=config)

        print(f"using {device_type}")
        model.to(device_type)

        checkpoint = torch.load("model_stage2.pth", map_location=device_type)
        model.load_state_dict(checkpoint)

        model = torch.compile(model)
        model.eval()

    loader = DataLoaderStage2(mode="val")

    prediction = []
    correct = torch.zeros((), dtype=torch.long, device=device_type)

    gru_hidden = None  # currently only used by stage1
    num_examples = 0

    while loader.not_empty:
        x, y = loader.next_batch()
        x, y = x.to(device_type), y.to(device_type)

        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # current implementation is tearcher forcing accuracy for stage2
                logits, _ = model(x, y, gru_hidden)  # (N-1, num_class)

            pred = logits.argmax(dim=1)  # (B,)
            prediction.append(pred)

            num_examples += len(y[1:])
            correct += (pred == y[1:]).sum()

    accuracy = (correct.float() / num_examples).item()

    out = torch.cat(prediction, dim=0).cpu()  # (num_of_examples,)
    out = out.numpy().astype(int)

    return accuracy, out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run eval")
    parser.add_argument(
        "stage", help="which eval stage to run, stage0 or stage1 or stage2"
    )

    args = parser.parse_args()
    accuracy, out = eval(args.stage)

    print(f"accuracy is {accuracy:.4f}")

    decode = [ITOSSTAGE2[i] for i in out]
    df = pd.DataFrame(decode, columns=["predictions"])
    df.to_csv("out_tensor_stage2.csv", index=False)
