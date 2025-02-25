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
    env_cfg, obs_cfg, camera_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    # camera_cfg = {
    #     "fixed_camera": {
    #         "res":(1280, 960),
    #         "pos":(10, 10, 10),
    #         "lookat":(0, 0, 0.5),
    #         "fov":30,
    #         "GUI":False,
    #         "use_depth": False
    #     },
    #     "follower_camera": {
    #         "res":(224, 224),
    #         "pos":(-1, 3.0, 2),
    #         "lookat":(0.0, 0.0, 0.5),
    #         "fov":30,
    #         "GUI":False,
    #         "use_depth": False
    #     },
    #     "head_camera": {
    #         "res":(224, 224),
    #         "pos":(0, 0, 0.5),
    #         "lookat":(0, 0, 0.5),
    #         "fov":30,
    #         "GUI":False,
    #         "use_rgb": False,
    #         "use_depth": True
    #     }
    # }
    camera_cfg["fixed_camera"]["use_depth"] = False
    camera_cfg["follower_camera"]["use_depth"] = False
    camera_cfg["head_camera"]["use_rgb"] = False
    camera_cfg["head_camera"]["use_depth"] = True
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        camera_cfg=camera_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _, img_obs = env.reset()
    # RGB, depth, segmentation, normal
    # rgb, depth, segmentation, normal = env.cam.render(depth=True, segmentation=True, normal=True)
    # env.cam.start_recording()
    env.fixed_camera.start_recording()
    env.follower_camera.start_recording()
    # env.head_camera.start_recording()
    for i in range(len(env.head_cameras)):
        env.head_cameras[i].start_recording()
    steps = 100
    s = 0
    with torch.no_grad():
        while True:
            # env.cam.render()
            actions = policy((obs.to("cuda:0"), img_obs.to("cuda:0")))
            obs, _, rews, dones, infos, img_obs = env.step(actions)
            s += 1
            # print(s)
            # if s >= steps:
            #     env.reset()
            #     print("reset")
            #     s = 0
            if s >= 2000:
                break
    # env.cam.stop_recording(save_to_filename='video.mp4', fps=60)
    env.fixed_camera.stop_recording(save_to_filename=os.path.join(log_dir, f'video_eval_ckpt_{args.ckpt}.mp4'), fps=60)
    env.follower_camera.stop_recording(save_to_filename=os.path.join(log_dir, f'follow_video_eval_ckpt_{args.ckpt}.mp4'), fps=60)
    # env.head_camera.stop_recording(save_to_filename='head_video_eval.mp4', fps=60)
    for i in range(len(env.head_cameras)):
        env.head_cameras[i].stop_recording(save_to_filename=os.path.join(log_dir, f'head_video_eval_{i}_ckpt_{args.ckpt}.mp4'), fps=60)

if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100
"""
