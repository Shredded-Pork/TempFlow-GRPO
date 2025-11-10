import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.num_epochs = 100
    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # # 8*A800
    # config.resolution = 512
    # config.sample.train_batch_size = 12
    # config.sample.num_image_per_prompt = 24
    # config.sample.num_batches_per_epoch = 12
    # config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    # 1 A800. This is just to ensure it runs quickly on a single GPU, though the performance may degrade.
    # If using 8 GPUs, please comment out this section and use the 8-GPU configuration above instead.
    config.resolution = 512
    config.sample.train_batch_size = 12
    config.sample.num_image_per_prompt = 6
    config.sample.num_batches_per_epoch = 6
    config.sample.test_batch_size = 16 

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.004
    # kl reward
    # KL reward and KL loss are two ways to incorporate KL divergence. KL reward adds KL to the reward, while KL loss, introduced by GRPO, directly adds KL loss to the policy loss. We support both methods, but KL loss is recommended as the preferred option.
    config.sample.kl_reward = 0
    # We also support using SFT data in RL training for supervised learning to prevent quality drop, but this option was unused
    config.train.sft=0.0
    config.train.sft_batch_size=3
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std=True
    config.train.ema=True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        # "geneval": 1.0,
        "ocr": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "your path"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # 8*A800
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.004
    config.sample.kl_reward = 0
    config.sample.global_std=False
    config.train.ema=True
    config.num_epochs = 3000
    config.save_freq = 100 # epoch
    config.eval_freq = 20
    config.save_dir = 'save dir'
    config.reward_fn = {
        "geneval": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    config.run_name = 'run name' 

    return config

def geneval_sd3_pr():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "your path"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # 8*A800
    config.resolution = 512
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.004
    config.sample.kl_reward = 0
    config.sample.global_std=False
    config.train.ema=True
    config.num_epochs = 3000
    config.save_freq = 100 # epoch
    config.eval_freq = 20
    config.save_dir = 'save dir'
    config.reward_fn = {
        "geneval": 1.0,
        # "imagereward": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    config.run_name = 'run name'

    return config

def pickscore_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "your path"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # 8*A800 keep group size=24 num_groups=48 & 1ep update 2
    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 16 # # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.001    
    config.train.sft=0.0
    config.train.sft_batch_size=3
    config.sample.kl_reward = 0
    config.sample.global_std=False
    config.train.ema=True
    config.num_epochs = 1000
    config.save_freq = 100 # epoch
    config.eval_freq = 20
    config.save_dir = 'save dir'
    config.reward_fn = {
        # "geneval": 1.0,
        "pickscore": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True

    config.run_name = 'run name'
    return config


def pickscore_sd3_pr():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "your path"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5

    # 8*A800 keep group size=24，num group=48
    config.resolution = 512
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 16 # # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size 
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.001    
    config.train.sft=0.0
    config.train.sft_batch_size=3
    config.sample.kl_reward = 0
    config.sample.global_std=False
    config.train.ema=True
    config.num_epochs = 1000
    config.save_freq = 100 # epoch
    config.eval_freq = 20
    config.save_dir = 'save dir'
    config.reward_fn = {
        # "geneval": 1.0,
        "pickscore": 1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True

    config.run_name = 'run name'
    return config


def flux():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/hpsv2")
    config.mixed_precision = 'bf16'


    # sd3.5 medium
    config.pretrained.model = "your path"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale=3.5

    # 8*A800 keep group size=24 num_groups=48 & 1ep update 2
    config.resolution = 1024
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 6 # # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.004    
    config.train.sft=0.0
    config.train.sft_batch_size=3
    config.sample.kl_reward = 0
    config.sample.global_std=False
    config.train.ema=True
    config.num_epochs = 250
    config.save_freq = 30 # epoch
    config.eval_freq = 20
    config.save_dir = 'save dir'
    config.reward_fn = {
        # "geneval": 1.0,
        # "pickscore": 1.0,
        # "unifiedreward": 0.7,
        # ICCV 2025 HPSv3
        "hpsv3": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True

    config.run_name = 'run name'
    return config


def flux_pr():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/hpsv2")
    config.mixed_precision = 'bf16'


    # sd3.5 medium
    config.pretrained.model = "your path "
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale=3.5

    # 8*A800 keep group size=24 num_groups=48 & 1ep update 2
    config.resolution = 1024
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 6 # # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.004    
    config.train.sft=0.0
    config.train.sft_batch_size=3
    config.sample.kl_reward = 0
    config.sample.global_std=False
    config.train.ema=True
    config.num_epochs = 250
    config.save_freq = 30 # epoch
    config.eval_freq = 20
    config.save_dir = 'save dir'
    config.reward_fn = {
        # "geneval": 1.0,
        # "pickscore": 1.0,
        # "unifiedreward": 0.7,
        # ICCV 2025 HPSv3
        "hpsv3": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True

    config.run_name = 'run name'
    return config

def pickscore_qwenimage_pr():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.mixed_precision = "bf16"

    # flux
    config.pretrained.model = "Qwen-Image"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = False
    config.train.ema = False
    config.num_epochs = 250
    config.save_freq = 30 # epoch
    config.eval_freq = 20
    config.save_dir = ''
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    config.run_name = ''

    return config

def get_config(name):
    return globals()[name]()
