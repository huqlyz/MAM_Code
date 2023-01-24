import numpy as np
import torch
import gym
import argparse
import os
import pandas
import utils
import MA_TD3
import OurDDPG
import DDPG
global evaluations1
global evaluations2
evaluations1 = []
evaluations2 = []
def assignment(policy, env_name, seed, eval_episodes, w1, w2, time, file_name):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 10)
    avg_reward1 = 0.
    avg_reward2 = 0.
    Tv1 = 0.
    Tv2 = 0.
    Ev1 = 0.
    Ev2 = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        Ev1 = Ev1 + policy.estimate_value(state, max_action, 1)
        n = 0
        while not done:
            action = policy.select_action1(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward1 += reward
            Tv1 = Tv1 + reward * (0.99 ** n)
            n += 1

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        n = 0
        action = policy.select_action1(np.array(state))
        Ev2 = Ev2 + policy.estimate_value(state, max_action, 2)
        while not done:
            action = policy.select_action2(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward2 += reward
            Tv2 = Tv2 + reward * (0.99 ** n)
            n += 1
            # eval_env.render()
    Ms = w1 + w2

    #  按奖励分配works
    avg1 = avg_reward1 / eval_episodes
    avg2 = avg_reward2 / eval_episodes
    evaluations1.append(avg1)
    evaluations2.append(avg2)
    Tv1 = Tv1 / eval_episodes
    Tv2 = Tv2 / eval_episodes
    Ev1 = Ev1 / eval_episodes
    Ev2 = Ev2 / eval_episodes
    Tv1 = float(Tv1)
    Tv2 = float(Tv2)
    Ev1 = float(Ev1)
    Ev2 = float(Ev2)

    result1.append([Tv1, Ev1, Tv1 - Ev1])
    result2.append([Tv2, Ev2, Tv2 - Ev2])

    if avg1 < avg2:
        if avg1 > 0:
            w1 = round(avg_reward1 / (avg_reward2 + avg_reward1) * Ms)
            w2 = Ms - w1
        elif avg2 < 0:
            w1 = round(avg_reward2 / (avg_reward2 + avg_reward1) * Ms)
            w2 = Ms - w1
        else:
            w1 = 0
            w2 = Ms
    else:
        if avg2 > 0:
            w1 = round(avg_reward1 / (avg_reward2 + avg_reward1) * Ms)
            w2 = Ms - w1
        elif avg1 < 0:
            w1 = round(avg_reward2 / (avg_reward2 + avg_reward1) * Ms)
            w2 = Ms - w1
        else:
            w1 = Ms
            w2 = 0

    if w1 == 0:
        if t < 300000:
            w1 = 1
            w2 = Ms - w1

    if w2 == 0:
        if t < 300000:
            w2 = 1
            w1 = Ms - w2

    np.save(f"./result/{file_name}a1", evaluations1)
    np.save(f"./result/{file_name}a2", evaluations2)
    print("---------------------------------------------------------------------------------")
    print(f"---------权重调整为 actor1: {w1}    actor2:{w2}   -------------------------------")
    print(f"actor1得分：{avg_reward1 / eval_episodes}   actor2得分：{avg_reward2 / eval_episodes} ")
    print("-----------------------------------------------------------------------------------")
    print(f"Critic1真实值：{Tv1} ----------Critic2估值{Ev1} -------------差值{Tv1 - Ev1} ")
    print(f"Critic2真实值：{Tv2} ----------Critic2估值{Ev2} -------------差值{Tv2 - Ev2} ")
    print("-----------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------")

    return w1, w2


def assignment_final(policy, env_name, seed, eval_episodes):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 10)
    avg_reward1 = 0.
    avg_reward2 = 0.

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        while not done:
            action = policy.select_action1(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward1 += reward

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action2(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward2 += reward
    f = 0
    if avg_reward1 > avg_reward2:
        f = 1
        print('-------------------------淘汰actor2,保留actor1------------------------')
        print('-------------------------淘汰actor2,保留actor1------------------------')
        print('-------------------------淘汰actor2,保留actor1------------------------')
    else:
        f = 2
        print('-------------------------淘汰actor1,保留actor2------------------------')
        print('-------------------------淘汰actor1,保留actor2------------------------')
        print('-------------------------淘汰actor1,保留actor2------------------------')

    return f


def evalpolicy(policy, env_name, seed, eval_episodes, f, time,file_name):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 10)
    avg_reward = 0.
    Ev = 0.
    Tv = 0.

    if f == 1:
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            n = 0
            Ev = Ev + policy.estimate_value(state, max_action, 1)
            while not done:
                action = policy.select_action1(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
                Tv = Tv + reward * (0.99 ** n)
                n += 1
        evaluations1.append(avg_reward / eval_episodes)
        np.save(f"./result/{file_name}a1", evaluations1)
    elif f == 2:
        for _ in range(eval_episodes):
            state, done = eval_env.reset(), False
            n = 0
            Ev = Ev + policy.estimate_value(state, max_action, 1)
            while not done:
                action = policy.select_action2(np.array(state))
                state, reward, done, _ = eval_env.step(action)
                avg_reward += reward
                Tv = Tv + reward * (0.99 ** n)
                n += 1
        evaluations2.append(avg_reward / eval_episodes)
        np.save(f"./result/{file_name}a2", evaluations2)

    avg = avg_reward / eval_episodes
    Ev = Ev / eval_episodes
    Tv = Tv / eval_episodes
    Tv = float(Tv)
    Ev = float(Ev)
    result.append([Tv, Ev, Tv-Ev])
    # print(result)
    print(f"-----------------------------------------------------------------")
    print(f"---------当前 actor{f}得分为{avg}   -------------------------------")
    print(f"---------当前 actor{f}得分为{avg}   -------------------------------")
    print(f"---------当前 critic{f}估计值{Ev}   -------------------------------")
    print(f"---------当前 critic{f}真实值{Tv}   -------------------------------")
    print(f"-----------------------------------------------------------------")




def eval_policy(policy, env_name, seed, eval_episodes, file_name):

    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward1 = 0.
    avg_reward2 = 0.
    Tv1 = 0.
    Tv2 = 0.
    Ev1 = 0.
    Ev2 = 0.

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False

        Ev1 = Ev1 + policy.estimate_value(state, max_action, 1)
        n = 0
        while not done:
            action = policy.select_action1(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward1 += reward
            Tv1 = Tv1 + reward * (0.99 ** n)
            n += 1

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        n = 0
        action = policy.select_action1(np.array(state))
        Ev2 = Ev2 + policy.estimate_value(state, max_action, 2)
        while not done:
            action = policy.select_action2(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward2 += reward
            Tv2 = Tv2 + reward * (0.99 ** n)
            n += 1

    avg1 = avg_reward1 / eval_episodes
    avg2 = avg_reward2 / eval_episodes
    evaluations1.append(avg1)
    evaluations2.append(avg2)
    Tv1 = Tv1 / eval_episodes
    Tv2 = Tv2 / eval_episodes
    Ev1 = Ev1 / eval_episodes
    Ev2 = Ev2 / eval_episodes
    Tv1 = float(Tv1)
    Tv2 = float(Tv2)
    Ev1 = float(Ev1)
    Ev2 = float(Ev2)

    result1.append([Tv1, Ev1, Tv1 - Ev1])
    result2.append([Tv2, Ev2, Tv2 - Ev2])

    np.save(f"./result/{file_name}a1", evaluations1)
    np.save(f"./result/{file_name}a2", evaluations2)
    print("---------------------------------------------------------------------------------")
    print(f"actor1得分：{avg_reward1 / eval_episodes}   actor2得分：{avg_reward2 / eval_episodes} ")
    print("-----------------------------------------------------------------------------------")
    print(f"Critic1真实值：{Tv1} ----------Critic1估值{Ev1} -------------差值{Tv1 - Ev1} ")
    print(f"Critic2真实值：{Tv2} ----------Critic2估值{Ev2} -------------差值{Tv2 - Ev2} ")
    print("-----------------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------------")

    return 0

if __name__ == "__main__":
    print('ant 不同折扣  30万步前拷贝参数 1e6 256')

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="MA_TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="Ant-v2")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--M", default=3, type=int)  # M
    parser.add_argument("--assign", default=5e3, type=int)  # How often (time steps) we Assign weights for generating experience
    parser.add_argument("--final_step", default=3e5,
                        type=int)  # elimate poor actor
    parser.add_argument("--eta", default=0.1, type=float)

    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./result"):
        os.makedirs("./result")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    filename = './result/' + f"{args.policy}_{args.env}_{args.seed}" + '.txt'
    ######################################################

    # creat_envs(args.M)
    env = [gym.make(args.env) for i in range(args.M)]

    # Set seeds
    for i in range(args.M):
        env[i].seed(args.seed)
        env[i].action_space.seed(args.seed)
        print(env[i])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env[0].observation_space.shape[0]

    action_dim = env[0].action_space.shape[0]
    max_action = float(env[0].action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "MA_TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = MA_TD3.MA_TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    # evaluations = [eval_policy(policy, args.env, args.seed)]
    state = [env[i].reset() for i in range(args.M)]
    done = [False for i in range(args.M)]
    episode_reward = [0 for i in range(args.M)]
    episode_timesteps = [0 for i in range(args.M)]
    episode_num = [0 for i in range(args.M)]
    global result  # 保存淘汰之后的估值信息
    global result1  # 保存actor1的估值信息
    global result2  # 保存actor2的估值信息
    result = []
    result1 = []
    result2 = []
    f = 0 # 标记挑选后的actor下标
    for t in range(int(args.max_timesteps)):
        if t % 1e4 == 0:
            print("------------------------------------------")
            print(f"当前step---------------{t}----------------")
            print("------------------------------------------")
        for i in range(args.M):
            episode_timesteps[i] += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env[0].action_space.sample()

            w1 = int(args.M / 2)
            w2 = int(args.M - w1)

            next_state, reward, done[0], _ = env[0].step(action)
            done_bool = float(done[0]) if episode_timesteps[0] < env[0]._max_episode_steps else 0

            # Store data in replay buffer

            replay_buffer.add(state[0], action, next_state, reward, done_bool)

            state[0] = next_state
            episode_reward[0] += reward

            if t % args.assign == 0:
                eval_policy(policy, args.env, args.seed,10,file_name)

        else:  # 修改动作机制
            if t % args.assign == 0 and t > args.start_timesteps:  # reassgin weight   #  # assign weight for generate experience
                if f == 0:
                    t1 = w1
                    t2 = w2
                    w1, w2 = assignment(policy, args.env, args.seed, 15, t1, t2, t, file_name)
                    w1 = int(w1)
                    w2 = int(w2)
                else:
                    evalpolicy(policy, args.env, args.seed, 10, f, t,file_name)
                    if f == 1:
                        w1 = 1
                        w2 = 0
                    elif f == 2:
                        w1 = 0
                        w2 = 1

            for i in range((w1)):  # actor1 generate exp
            #  print(f'当前M编号{i}')
                action = (
                    policy.select_action1(np.array(state[i]))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

                next_state, reward, done[i], _ = env[i].step(action)
                done_bool = float(done[i]) if episode_timesteps[i] < env[i]._max_episode_steps else 0
                replay_buffer.add(state[i], action, next_state, reward, done_bool)
                state[i] = next_state
                episode_reward[i] += reward

            for j in range(int(w2)):  # actor1 generate exp
                i = j + w1
            # print(f'当前M编号{i}')
                action = (
                    policy.select_action2(np.array(state[i]))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

                next_state, reward, done[i], _ = env[i].step(action)
                done_bool = float(done[i]) if episode_timesteps[i] < env[i]._max_episode_steps else 0
                replay_buffer.add(state[i], action, next_state, reward, done_bool)
                state[i] = next_state
                episode_reward[i] += reward

        # Train agent after collecting sufficient data

        if t >= args.start_timesteps and t <= args.final_step and t % 2 == 0:
            policy.train(replay_buffer, w1, w2, args.batch_size,args.eta, 1)  #
            policy.train(replay_buffer, w1, w2, args.batch_size,args.eta, 2)  #
            # policy.train(replay_buffer, w1, w2, args.batch_size, 1, 1)  # 延迟更新actor1 target_critic1
            # policy.train(replay_buffer, w1, w2, args.batch_size, 2, 2)  # 延迟更新actor2 target_critic1

        if t >= args.start_timesteps and t <= args.final_step and t % 2 == 1:
            policy.train(replay_buffer, w1, w2, args.batch_size, args.eta, 1,0)  #
            policy.train(replay_buffer, w1, w2, args.batch_size, args.eta, 2,0)  #
            policy.train(replay_buffer, w1, w2, args.batch_size, args.eta, 1, 1)  # 延迟更新actor1 target_critic1
            policy.train(replay_buffer, w1, w2, args.batch_size, args.eat, 2, 2)  # 延迟更新actor2 target_critic1

        if t >  args.final_step and f == 0:
            f = assignment_final(policy, args.env, args.seed, 40)
        elif f == 1:
            if t % 2 == 0:
                k = 0
                policy.train(replay_buffer, w1, w2, args.batch_size, args.eta, 1, 0)
            elif  t % 2 == 1:
                k = 0
                policy.train(replay_buffer, w1, w2, args.batch_size, args.eta, 1, 0)
                policy.train(replay_buffer, w1, w2, args.batch_size, args.eta, 1, 1)


        elif f == 2:

            if t % 2 == 0:

                policy.train(replay_buffer, w1, w2, args.batch_size, args.eta, 2, 0)

            elif t % 2 == 1:

                policy.train(replay_buffer, w1, w2, args.batch_size, args.eta, 2, 0)

                policy.train(replay_buffer, w1, w2, args.batch_size, args.eta, 2, 2)

        for i in range(args.M):
            if f != 0:
                i = 0
            if done[i]:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # print(f"当前权重：actor1 0-{w1}  actor2 {w1 + 1}-{w2 + w1}")
                print(f'当前M编号{i}')
                print(
                    f"Total T: {t + 1} Episode Num: {episode_num[i] + 1} Episode T: {episode_timesteps[i]} Reward: {episode_reward[i]:.3f}")
                # Reset environment
                state[i], done[i] = env[i].reset(), False
                episode_reward[i] = 0
                episode_timesteps[i] = 0
                episode_num[i] += 1

    name = ['tvalue', 'evalue', 'rate']
    if f == 1 :
        result1.extend(result)
        dataFrame = pandas.DataFrame(columns=name, data=result1)
        filename = './result/' + f"{args.policy}_{args.env}_{args.seed}" + '.cvs'
        dataFrame.to_csv(filename)

    elif f == 2 :
        result2.extend(result)
        dataFrame = pandas.DataFrame(columns=name, data=result2)
        filename = './result/' + f"{args.policy}_{args.env}_{args.seed}" + '.cvs'
        dataFrame.to_csv(filename)
