# CourtAnalytic
## Description
- This is a continuation to the Court Analytic CNN RNN approach
- The only difference is that we replaced the top RNN encoder decoder layers with a transformer decoder
- Similiar to PaliGemma, we unmasked the prefix, the frame encodings, during attention, while the shot category predictions are lower triangular masked

## Reference
- The ConvNeXt was build on the implementation from [CovNeXt](https://github.com/facebookresearch/ConvNeXt).
- The training code took reference from [nanoGPT](https://github.com/karpathy/nanoGPT)
