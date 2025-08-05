# TempFlow-GRPO
<div style="text-align: center;">
    <img src="asset/fig1.svg" alt="LOGO">
</div>
<b>  </b>

Recent flow matching models for text-to-image generation have achieved remark-
able quality, yet their integration with reinforcement learning for human prefer-
ence alignment remains suboptimal, hindering fine-grained reward-based opti-
mization. We observe that the key impediment to effective GRPO training of flow
models is the temporal uniformity assumption in existing approaches: sparse ter-
minal rewards with uniform credit assignment fail to capture the varying criticality
of decisions across generation timesteps, resulting in inefficient exploration and
suboptimal convergence. To remedy this shortcoming, we introduce TempFlow-
GRPO (Temporal Flow GRPO), a principled GRPO framework that captures and
exploits the temporal structure inherent in flow-based generation. introduces two
key innovations: (i) a trajectory branching mechanism that provides process re-
wards by concentrating stochasticity at designated branching points, enabling pre-
cise credit assignment without requiring specialized intermediate reward models;
and (ii) an noise weighting scheme that modulates policy optimization according
to the intrinsic exploration potential of each timestep, prioritizing learning during
high-impact early stages while ensuring stable refinement in later phases. These
additions endow the model with temporally-aware optimization that respects the
underlying generative dynamics, translating into state-of-the-art gains in human
preference alignment, and standard text-to-image benchmarks.

## Curve
![Eval on PickScore](asset/)

## Visualization
![Visualization](asset/vis1.svg)
![Visualization](asset/vis2.svg)
