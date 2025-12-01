from Models.ConvNeXt import ConvNeXt
from Classifiers.Classifier import FrameClassifier, ShotPredictor
from Loaders.Dataloader import DataLoaderStage0, DataLoaderStage1, DataLoaderStage2
import torch
import numpy as np
import pandas as pd
import argparse
from Loaders.vocab import ITOSSTAGE1, ITOSSTAGE2

STAGE2 = True

def eval(stage, model = None):
    """ stage: "stage1", "stage2"
    """
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    
    if model is None:
        # conv  = ResNet_RS()
        if stage == "stage0":
            conv = ConvNeXt(stage0=True)
        else:
            conv = ConvNeXt()
        
        if stage == "stage0":
            model = FrameClassifier(conv, stage0=True)
        elif stage == "stage1":
            pass
        else: # stage2
            pass

        print(f"using {device_type}")
        model.to(device_type)
        # be careful torch.compile(model) returns a wrapper 
        # with all parameter keys get prefixed with "_orig_mod."
        # so we have to torch.compile before load_state_dict to match keys
        # think similar issue happen with DDP as well

        if stage == "stage0":
            checkpoint = torch.load("model_stage0.pth", map_location = device_type)
            model.load_state_dict(checkpoint)
        elif stage == "stage1":
            checkpoint = torch.load("model_stage1.pth", map_location = device_type)
            model.load_state_dict(checkpoint)
        else:
            checkpoint = torch.load("model_stage2.pth", map_location = device_type)
            model.load_state_dict(checkpoint)

        model = torch.compile(model)
        model.eval()

    if stage == "stage0":
        loader = DataLoaderStage0(mode="val")
    elif stage == "stage1":
        loader = DataLoaderStage1(mode="val") 
    else:
        loader = DataLoaderStage2(mode="val") 

    prediction = []
    correct = torch.zeros((), dtype=torch.long, device=device_type)

    gru_hidden = None # currently only used by stage1
    num_examples = 0

    while loader.not_empty:
        x, y = loader.next_batch()
        x, y = x.to(device_type), y.to(device_type)

        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # current implementation is tearcher forcing accuracy for stage2
                logits, _ = model(x, y, gru_hidden) # (N-1, num_class)

            pred = logits.argmax(dim = 1) # (B,)
            prediction.append(pred)
            
            num_examples += len(y[1:])
            correct += (pred == y[1:]).sum()

    accuracy = (correct.float() / num_examples).item()

    out = torch.cat(prediction, dim=0).cpu() # (num_of_examples,)
    out = out.numpy().astype(int)

    return accuracy, out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run eval")
    parser.add_argument("stage", help="which eval stage to run, stage0 or stage1 or stage2")

    args = parser.parse_args()
    accuracy, out = eval(args.stage)

    print(f"accuracy is {accuracy:.4f}")

    if args.stage == "stage2":
        decode = [ITOSSTAGE2[i] for i in out]
        df = pd.DataFrame(decode, columns=["predictions"])
        df.to_csv("out_tensor_stage2.csv", index=False)
    else: # stage0 and stage1
        decode = [ITOSSTAGE1[i] for i in out]
        df = pd.DataFrame(decode, columns=["predictions"])
        if args.stage == "stage0":
            df.to_csv("out_tensor_stage0.csv", index=False)
        else:
            df.to_csv("out_tensor_stage1.csv", index=False)


    




