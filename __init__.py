from .frame_segmenter import FrameSegmenter, FrameSegmenterIndexer
from .repeat_sampler import RepeatSamplerConfigNode, RepeatSampler, RepeatSamplerConfigPatchModel, RepeatSamplerConfigPatchLatent

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FrameSegmenter": FrameSegmenter,
    "FrameSegmenterIndexer": FrameSegmenterIndexer,
    "RepeatSamplerConfigNode": RepeatSamplerConfigNode,
    "RepeatSamplerConfigPatchModel": RepeatSamplerConfigPatchModel,
    "RepeatSamplerConfigPatchLatent": RepeatSamplerConfigPatchLatent,
    "RepeatSampler": RepeatSampler,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FrameSegmenter": "Frame Segmenter",
    "FrameSegmenterIndexer": "Get Frame at Index",
    "RepeatSamplerConfigNode": "Repeat Sampler Config",
    "RepeatSamplerConfigPatchModel": "Patch Repeat Sampler Config (Model)",
    "RepeatSamplerConfigPatchLatent": "Patch Repeat Sampler Config (Latent)",
    "RepeatSampler": "KSampler (Simple Input)",
}