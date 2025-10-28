<div align="center" style="font-family: charter;">

<h1>🦖 TEMPFLOW-GRPO: WHEN TIMING MATTERS FOR GRPO IN FLOW MODELS</h1>


<a href="https://arxiv.org/abs/2508.04324" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-TempFlowGRPO-red?logo=arxiv" height="20" /></a>
<a href="https://tempflowgrpo.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/💻_Project-TempFlowGRPO-blue.svg" height="20" /></a>
</div>

**TempFlow-GRPO** (Temporal Flow GRPO), a principled GRPO framework that captures and exploits the temporal structure inherent in flow-based generation. 

<div style="text-align: center;">
    <img src="asset/figure3.jpg" alt="LOGO">
</div>

<div style="text-align: center;">
    <img src="asset/teaser.png" alt="LOGO">
</div>

## 🗺️ Roadmap for TempFlow-GRPO
> TempFlow-GRPO (Temporal Flow GRPO), a principled GRPO framework that captures and exploits the temporal structure inherent in flow-based generation. TempFlow-GRPO introduces two key innovations: (i) a trajectory branching mechanism that provides process rewards by concentrating stochasticity at designated branching points, enabling precise credit assignment without requiring specialized intermediate reward models; and (ii) a noise-aware weighting scheme that modulates policy optimization according to the intrinsic exploration potential of each timestep, prioritizing learning during high-impact early stages while ensuring stable refinement in later phases. These innovations endow the model with temporally-aware optimization that respects the underlying generative dynamics, leading to state-of-the-art performance in human preference alignment and standard text-to-image benchmark.
> 
> Welcome Ideas and Contributions. Stay tuned!

## 🆕 News

> We have presented an improved **Flow-GRPO** method, **TempFlow-GRPO**. We will release our code recently!🔥🔥🔥
- **[2025-08-06]**  We have released the first version of our paper. 🔥🔥🔥
- **[2025-08-11]**  Thanks [Jie Liu's](https://jieliu.site/) comments for our paper. We will release the 1024 Flux RL model in the month. 🔥🔥🔥
- **[2025-08-14]**  Our method also achieves better performance in FLUX 1024px with HPSv3 (based on Qwen2-VL) as reward. 🔥🔥🔥
- **[2025-08-20]**  We have released the first version of our paper in huggface. 🔥🔥🔥
- **[2025-09-12]**  We will release the second version of our paper in next week. 🔥🔥🔥
- **[2025-09-17]**  We will release the code of our paper. 🔥🔥🔥
- **[2025-10-28]**  Very happy to see TempFlow-GRPO in video RL of meituan's [Longcat-Video](https://github.com/meituan-longcat/LongCat-Video). 🔥🔥🔥
- **[2025-10-28]**  Very happy to see TempFlow-GRPO in image edit RL of baai's [OmniGen2-EditScore](https://github.com/VectorSpaceLab/EditScore). 🔥🔥🔥




## 🚀 Updates
To support research and the open-source community, we will release the entire project—including datasets, training pipelines, and model weights. **Our code is based on Flow-GRPO!**. Thank you for your patience and continued support! 🌟
- [x] Release arXiv paper
- [x] Release GitHub repo
- [x] Release training code
- [ ] Release neat training code
- [ ] Release model checkpoints

## 📕 Training & Evaluation
### Preparation
1. First you need to download the reward model (we support clip-based pickscore, vlm-based hpsv3, ...) and base model (SD3.5-M, FLUX.1-dev).
2. Then you need to modify the noise level in [sd3_pipeline_with_logprob_perstep](https://github.com/Shredded-Pork/TempFlow-GRPO/blob/main/flow_grpo/diffusers_patch/sd3_pipeline_with_logprob_perstep.py) and [sd3_pipeline_with_logprob](https://github.com/Shredded-Pork/TempFlow-GRPO/blob/main/flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py).
3. Finally, you need to modify the [config](https://github.com/Shredded-Pork/TempFlow-GRPO/blob/main/config/dgx.py). We suggest you using 24 groups and 48 num groups.

**Note that we use branch=4, per branch exploration=6. You can modify them in our code. We will release a neat code verision in next few days.**

### Training
#### About Group Strategy
1. Seed Group:
<img width="1011" height="45" alt="image" src="https://github.com/user-attachments/assets/e865d9db-3f8f-45a7-a497-2298b0013a42" />
<img width="894" height="41" alt="image" src="https://github.com/user-attachments/assets/9953b278-3c91-4b27-8ac5-f67c0199b5f4" />
<img width="892" height="119" alt="image" src="https://github.com/user-attachments/assets/06771eb4-6974-4ddf-a02e-2ec29b3382c4" />

2. Prompt Group: notes the seed group

3. Batch Group: global_std=True
   
#### SD3.5-M
```bash
# Flow-GRPO
bash scripts/multi_node/main.sh
# TempFlow-GRPO
bash scripts/multi_node/train_sd3_pr.sh
```
#### FLUX.1-dev
```bash
# Flow-GRPO
bash scripts/multi_node/train_flux.sh
# TempFlow-GRPO
bash scripts/multi_node/train_flux_pr.sh
```


## 📊 Experimental Performance
<img src="asset/performance.png" alt="Performance" width="800"/>

## 📺 Visualization
<img src="asset/appendix_vis4.png" alt="FLUX.1-dev" width="1024"/> 

- For more details please read our paper.

# Acknowledgements
[Flow-GRPO](https://github.com/yifan123/flow_grpo): The first method integrating online reinforcement learning (RL) into flow matching models.
