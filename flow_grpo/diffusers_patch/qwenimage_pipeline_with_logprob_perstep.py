from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift
from .sd3_sde_with_logprob_perstep import sde_step_with_logprob

@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    true_cfg_scale: float = 4.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: Optional[float] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    determistic: bool = True,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )

    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        prompt=prompt+negative_prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    prompt_embeds, negative_prompt_embeds = prompt_embeds.chunk(2, dim=0)
    prompt_embeds_mask, negative_prompt_embeds_mask = prompt_embeds_mask.chunk(2, dim=0)
    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
        logger.warning(
            f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
        )
        guidance = None
    elif not self.transformer.config.guidance_embeds and guidance_scale is None:
        guidance = None
    if self.attention_kwargs is None:
        self._attention_kwargs = {}

    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
    negative_txt_seq_lens = (
        negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
    )

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_sde_latents = []
    all_log_probs = []
    all_image = []
    all_timesteps = []

    # 7. Denoising loop
    self.scheduler.set_begin_index(0)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            noise_pred = self.transformer(
                hidden_states=torch.cat([latents, latents], dim=0),
                timestep=torch.cat([timestep, timestep], dim=0) / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=torch.cat([prompt_embeds_mask, negative_prompt_embeds_mask], dim=0),
                encoder_hidden_states=torch.cat([prompt_embeds, negative_prompt_embeds], dim=0),
                img_shapes=img_shapes*2,
                txt_seq_lens=txt_seq_lens+negative_txt_seq_lens,
            )[0]
            noise_pred, neg_noise_pred = noise_pred.chunk(2, dim=0)
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)
            latents_dtype = latents.dtype

            # ODE
            ode_latents, _, _, _ = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                t.unsqueeze(0).repeat(latents.shape[0]),
                latents.float(),
                determistic=True,
            )
            # SDE
            sde_latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler,
                noise_pred.float(),
                t.unsqueeze(0),
                latents.float(),
                determistic=False,
            )
            latents = ode_latents
            prev_latents = latents.clone()

            all_latents.append(latents)
            all_sde_latents.append(sde_latents)
            all_log_probs.append(log_prob)
            all_timesteps.append(t)

            # SDE sample branch, then ode process
            inner_latents = sde_latents.clone()
            for inner_i, inner_t in enumerate(timesteps[i+1:]):
                inner_timestep = inner_t.expand(inner_latents.shape[0]).to(inner_latents.dtype)
                inner_noise_pred = self.transformer(
                    hidden_states=torch.cat([inner_latents, inner_latents], dim=0),
                    timestep=torch.cat([inner_timestep, inner_timestep], dim=0) / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=torch.cat([
                        prompt_embeds_mask.repeat_interleave(inner_latents.shape[0]//prompt_embeds.shape[0], dim=0),
                        negative_prompt_embeds_mask.repeat_interleave(inner_latents.shape[0]//prompt_embeds.shape[0], dim=0)
                    ], dim=0),
                    encoder_hidden_states=torch.cat([
                        prompt_embeds.repeat_interleave(inner_latents.shape[0]//prompt_embeds.shape[0], dim=0),
                        negative_prompt_embeds.repeat_interleave(inner_latents.shape[0]//prompt_embeds.shape[0], dim=0)
                    ], dim=0),
                    img_shapes=img_shapes*2,
                    txt_seq_lens=txt_seq_lens+negative_txt_seq_lens,
                )[0]
                inner_noise_pred, inner_neg_noise_pred = inner_noise_pred.chunk(2, dim=0)
                inner_comb_pred = inner_neg_noise_pred + true_cfg_scale * (inner_noise_pred - inner_neg_noise_pred)

                inner_cond_norm = torch.norm(inner_noise_pred, dim=-1, keepdim=True)
                inner_noise_norm = torch.norm(inner_comb_pred, dim=-1, keepdim=True)
                inner_noise_pred = inner_comb_pred * (inner_cond_norm / inner_noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                inner_latents_dtype = inner_latents.dtype
                inner_latents, _, _, _ = sde_step_with_logprob(
                    self.scheduler,
                    inner_noise_pred.float(),
                    inner_t.unsqueeze(0),
                    inner_latents.float(),
                    determistic=True,
                )
                if inner_latents.dtype != inner_latents_dtype:
                    inner_latents = inner_latents.to(inner_latents_dtype)

            inner_latents = self._unpack_latents(inner_latents, height, width, self.vae_scale_factor)
            inner_latents = inner_latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(inner_latents.device, inner_latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                inner_latents.device, inner_latents.dtype
            )
            inner_latents = inner_latents / latents_std + latents_mean
            inner_image = self.vae.decode(inner_latents, return_dict=False)[0][:, :, 0]
            inner_image = self.image_processor.postprocess(inner_image, output_type=output_type)
            all_image.append(inner_image)

            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = latents.to(self.vae.dtype)
    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    return all_image, all_latents, all_sde_latents, all_log_probs, all_timesteps, prompt_embeds, negative_prompt_embeds, prompt_embeds_mask, negative_prompt_embeds_mask
