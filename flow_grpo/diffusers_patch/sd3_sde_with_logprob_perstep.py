# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/flow_grpo/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput, is_scipy_available, logging
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput, FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    determistic: bool = False,
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
    # Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    # process from the learned model outputs (most often the predicted velocity).
    # Args:
    #     model_output (`torch.FloatTensor`):
    #         The direct output from learned flow model.
    #     timestep (`float`):
    #         The current discrete timestep in the diffusion chain.
    #     sample (`torch.FloatTensor`):
    #         A current instance of a sample created by the diffusion process.
    #     generator (`torch.Generator`, *optional*):
    #         A random number generator.
    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    # flux
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    # sd3.5-M
    # sigma = self.sigmas[step_index].view(-1, 1, 1, 1)
    # sigma_prev = self.sigmas[prev_step_index].view(-1, 1, 1, 1)
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*0.7 # In SD3.5-M, 0.7. In FLUX, 0.9
    
    # our sde
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    
    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )


    # No noise is added during evaluation
    if determistic:
        prev_sample = sample + dt * model_output
    else:
        num_samples = 6
        batch_size = sample.shape[0]
        
        prev_sample_mean = prev_sample_mean.repeat_interleave(num_samples, dim=0)
        
        noise_shape = [batch_size * num_samples] + list(model_output.shape[1:])
        variance_noise = randn_tensor(
            noise_shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    # 原始论文中的情况
    return prev_sample, log_prob, prev_sample_mean, std_dev_t * torch.sqrt(-1*dt)
    # 作者推荐修改的情况 此时beta*10
    # return prev_sample, log_prob, prev_sample_mean, std_dev_t
    