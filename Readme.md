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
- **[2025-08-14]**  Our method also achieves better performance in FLUX 1024px with HPSv3 (based on Qwen2-VL) as reward, blue is TempFlow-GRPO and Purple is Flow-GRPO Fixed. 🔥🔥🔥
<img width="420" height="203" alt="image" src="https://github.com/user-attachments/assets/fe2518c3-cd01-42d6-9564-8a357659a587" />

<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/c41b4873-0a43-42a1-a3a4-f4510be1c40a" /> <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/e33cbbff-c4e6-404b-a45e-01bb78f37522" /> <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/c804a8ba-b819-4ead-ab72-296485bad88f" /> <img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/00bf85e1-4049-4060-8845-696eb9fb9a3a" />

- **[2025-08-20]**  We have released the first version of our paper in huggface. 🔥🔥🔥
- **[2025-09-12]**  We will release the second version of our paper in next week. 🔥🔥🔥





## 🚀 Updates
To support research and the open-source community, we will release the entire project—including datasets, training pipelines, and model weights. Thank you for your patience and continued support! 🌟
- [x] Release arXiv paper
- [x] Release GitHub repo
- [x] Release training code
- [ ] Release model checkpoints

## 📊 Experimental Performance
<img src="asset/figure2_1.jpg" alt="PickScore" width="400"/> <img src="asset/figure2.jpg" alt="Geneval" width="400"/>

## 📺 Visualization
<img src="asset/figure7.jpg" alt="PickScore" width="1024"/> 

- For more details please read our paper.

# Acknowledgements
[Flow-GRPO](https://github.com/yifan123/flow_grpo): The first method integrating online reinforcement learning (RL) into flow matching models.
