import argparse
import os
import pickle

import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    # RGB, depth, segmentation, normal
    # rgb, depth, segmentation, normal = env.cam.render(depth=True, segmentation=True, normal=True)
    # env.cam.start_recording()
    env.follower_camera.start_recording()
    env.head_camera.start_recording()
    steps = 100
    s = 0
    with torch.no_grad():
        while True:
            # env.cam.render()
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            s += 1
            print(s)
            # if s >= steps:
            #     env.reset()
            #     print("reset")
            #     s = 0
            if s >= 300:
                break
    # env.cam.stop_recording(save_to_filename='video.mp4', fps=60)
    env.follower_camera.stop_recording(save_to_filename='follow_video.mp4', fps=60)
    env.head_camera.stop_recording(save_to_filename='head_video.mp4', fps=60)

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
