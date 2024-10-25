import comfy
import latent_preview
from nodes import common_ksampler, VAEDecode, VAEEncode

class RepeatSamplerConfig:
    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, latent_image, vae, denoise=1.0):
        self.model = model
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.latent_image = latent_image
        self.vae = vae
        self.denoise = denoise

    def __str__(self):
        return f"{self.model} {self.seed} {self.steps} {self.cfg} {self.sampler_name} {self.scheduler} {self.latent_image} {self.vae} {self.denoise}"

class RepeatSamplerConfigNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "vae": ("VAE", {"tooltip": "Optional VAE."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("SAMPLER_CONFIG",)
    FUNCTION = "create_config"
    CATEGORY = "AharaNodes/sampling"
    DESCRIPTION = "Creates a reuseable sampler config."
    def create_config(self, model, seed, steps, cfg, sampler_name, scheduler, latent_image, vae, denoise=1.0):
        return (RepeatSamplerConfig(model, seed, steps, cfg, sampler_name, scheduler, latent_image, vae, denoise=denoise), )


class RepeatSamplerConfigPatchModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "config": ("SAMPLER_CONFIG", {}),
            }
        }

    RETURN_TYPES = ("SAMPLER_CONFIG",)
    FUNCTION = "create_config"
    CATEGORY = "AharaNodes/sampling"
    DESCRIPTION = "Creates a reuseable sampler config."
    def create_config(self, model, config):
        config.model = model
        return (config, )

class RepeatSamplerConfigPatchLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "config": ("SAMPLER_CONFIG", {}),
            }
        }

    RETURN_TYPES = ("SAMPLER_CONFIG",)
    FUNCTION = "create_config"
    CATEGORY = "AharaNodes/sampling"
    DESCRIPTION = "Creates a reuseable sampler config."
    def create_config(self, latent_image, config):
        config.latent_image = latent_image
        return (config, )

class RepeatSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "config": ("SAMPLER_CONFIG", {}),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "negative": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to exclude from the image."}),
                "overwrite_denoise": ("BOOLEAN", {"default": False}),
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }, "optional": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", )
    FUNCTION = "sample"
    CATEGORY = "AharaNodes/sampling"
    DESCRIPTION = "Uses the provided model, config, positive and negative conditioning to denoise the latent image."

    def sample(self, config, positive, negative, overwrite_denoise, denoise=None, model=None, latent_image=None):
        denoise = denoise if overwrite_denoise else config.denoise
        model = model if model is not None else config.model
        latent_image = latent_image if latent_image is not None else config.latent_image

        #print(config)
        latent = common_ksampler(model, config.seed, config.steps, config.cfg, config.sampler_name, config.scheduler, positive, negative, latent_image, denoise=denoise)[0]
        latent["noise_mask"] = None

        if config.vae is not None:
            output_image = config.vae.decode(latent["samples"])
        else:
            output_image = None
        return (latent, output_image, )
