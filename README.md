<img width="378" height="192" alt="스크린샷 2025-10-26 오후 11 46 25" src="https://github.com/user-attachments/assets/1bc9e861-afd7-4414-9d10-da6a3453692e" />
evaluation_local***.py (주석)
evaluation_local.py
파일 2개 있는데
둘 다 로직은 같고
결과값 출력만 다릅니다

## 🧠 전체 코드

```C

import numpy as np
import random
import argparse
import os
from tabulate import tabulate
import sys
import csv

# ----------------------------------------------------------------------
# ⚙️ Dummy 설정 (환경/에이전트 없는 로컬 테스트용)
# ----------------------------------------------------------------------

class DummyRLAgent:
    """rl_agent가 없을 때를 대비한 더미 클래스. 랜덤 행동을 반환."""
    def choose_action(self, obs):
        return random.randint(0, 35)

try:
    from agents.rl.submission import agent as rl_agent
except ImportError:
    rl_agent = DummyRLAgent()

def make(env_type, conf=None, seed=1):
    print(f"\n❌ FATAL ERROR: Environment '{env_type}' not found.")
    print("Check 'env/chooseenv.py' or your project structure.")
    sys.exit(1)

try:
    from env.chooseenv import make
except ImportError:
    pass

# ----------------------------------------------------------------------
# 🧭 Action Map (힘 -100~200, 각도 -30~30)
# ----------------------------------------------------------------------
ACTION_MAP = {
    0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30],
    6: [-40, -30], 7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30],
    12: [20, -30], 13: [20, -18], 14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30],
    18: [80, -30], 19: [80, -18], 20: [80, -6], 21: [80, 6], 22: [80, 18], 23: [80, 30],
    24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6], 28: [140, 18], 29: [140, 30],
    30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18], 35: [200, 30]
}

RENDER = False

# ----------------------------------------------------------------------
# 🎮 행동 선택
# ----------------------------------------------------------------------
def get_joint_actions(state, algo_list):
    joint_actions = []
    for agent_idx, algo in enumerate(algo_list):
        if algo == 'random':
            force = float(random.uniform(-100, 200))
            angle = float(random.uniform(-30, 30))
        elif algo == 'rl':
            try:
                obs = state[agent_idx]['obs'].flatten()
                actions_raw = rl_agent.choose_action(obs)
                if isinstance(actions_raw, (int, np.integer)):
                    force, angle = ACTION_MAP[actions_raw]
                else:
                    force, angle = actions_raw[0], actions_raw[1]
            except Exception:
                force, angle = 0, 0
        else:
            force, angle = 0, 0

        force = float(np.clip(force, -100, 200))
        angle = float(np.clip(angle, -30, 30))
        joint_actions.append([[force], [angle]])
    return joint_actions

# ----------------------------------------------------------------------
# 🧱 환경 제어 / 보상 조정
# ----------------------------------------------------------------------
def restrict_zone_env(state, joint_action, agent_idx, zone_bounds):
    x_min, x_max, y_min, y_max = zone_bounds
    try:
        pos = state[agent_idx]['position']
        if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
            joint_action[agent_idx][0] = [0]
            joint_action[agent_idx][1] = [0]
    except Exception:
        pass
    return joint_action

def reward_zone_bonus(state, reward, agent_idx, zone_bounds, bonus_value=1.0):
    x_min, x_max, y_min, y_max = zone_bounds
    try:
        pos = state[agent_idx]['position']
        if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
            reward[agent_idx] += bonus_value
    except Exception:
        pass
    return reward

# ----------------------------------------------------------------------
# 💥 보상 함수
# ----------------------------------------------------------------------
def smoothness_reward(prev_action, curr_action):
    if prev_action is None or curr_action is None:
        return 0.0
    d_angle = abs(curr_action[1] - prev_action[1])
    d_force = abs(curr_action[0] - prev_action[0])
    smooth = 1.0 - 0.015 * d_angle - 0.002 * d_force  # 강화된 감쇠
    return max(smooth, 0.0)

def collision_penalty(info):
    if isinstance(info, dict) and info.get("collision", False):
        return -10.0  # 감점 강화
    return 0.0

# ----------------------------------------------------------------------
# 🏁 메인 실행 루프
# ----------------------------------------------------------------------
def run_game(env, algo_list, episode, shuffle_map, map_num,
             env_control_config=None, reward_bonus_config=None, verbose=False):

    num_agents = len(algo_list)
    total_reward = np.zeros(num_agents, dtype=float)
    num_win = np.zeros(num_agents + 1, dtype=int)  # [A0 승, A1 승, 무승부]
    success_steps = [[] for _ in range(num_agents)]
    prev_action = None

    for i in range(1, int(episode) + 1):
        episode_reward = np.zeros(num_agents, dtype=float)
        state = env.reset(shuffle_map)

        if RENDER:
            env.env_core.render()

        step = 0
        while True:
            # 🚀 초반 5스텝 강제 직진 (모든 맵 공통)
            if step < 1:
                joint_action = []
                for _ in algo_list:
                    joint_action.append([[150.0], [0.0]])  # 힘=150, 각도=0 (직진)
            else:
                joint_action = get_joint_actions(state, algo_list)

            # 환경 제어
            if env_control_config:
                agent_idx = env_control_config['agent']
                bounds = env_control_config['bounds']
                joint_action = restrict_zone_env(state, joint_action, agent_idx, bounds)

            try:
                next_state, reward, done, _, info = env.step(joint_action)
            except Exception:
                done = True
                reward = [0.0] * num_agents
                info = {}

            reward = np.array(reward, dtype=float)

            # 추가 보상
            a0 = (joint_action[0][0][0], joint_action[0][1][0])
            reward[0] += smoothness_reward(prev_action, a0)
            reward[0] += collision_penalty(info)
            prev_action = a0

            if reward_bonus_config:
                agent_idx = reward_bonus_config['agent']
                bounds = reward_bonus_config['bounds']
                bonus = reward_bonus_config.get('bonus', 1.0)
                reward = reward_zone_bonus(next_state, reward, agent_idx, bounds, bonus)

            episode_reward += reward
            step += 1
            if done:
                if reward[0] == 100 and reward[1] != 100:
                    num_win[0] += 1
                    success_steps[0].append(step)
                elif reward[1] == 100 and reward[0] != 100:
                    num_win[1] += 1
                    success_steps[1].append(step)
                else:
                    num_win[2] += 1
                if not verbose:
                    print('.', end='')
                    if i % 50 == 0 or i == episode:
                        print()
                break
            state = next_state

        total_reward += episode_reward

    total_reward /= float(episode)
    avg_steps = [np.mean(s) if s else 0 for s in success_steps]

    # -----------------------------
    # 🧾 결과 콘솔 출력
    # -----------------------------
    print("\n" + "=" * 50)
    print(f"Map {map_num} Result in {episode} Episodes")
    print("=" * 50)

    header = ['Name', algo_list[0], algo_list[1]]
    data = [
        ['Average Score', np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
        ['Wins', num_win[0], num_win[1]],
        ['Draws', '-', num_win[2]],
        ['Avg Steps (Win)',
         np.round(avg_steps[0], 1) if avg_steps[0] else '-',
         np.round(avg_steps[1], 1) if avg_steps[1] else '-']
    ]
    print(tabulate(data, headers=header, tablefmt='fancy_grid'))

    # -----------------------------
    # 💾 결과 CSV 자동 저장
    # -----------------------------
    result_path = "results.csv"
    file_exists = os.path.exists(result_path)
    with open(result_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Map", "Agent0", "Agent1", "Score0", "Score1",
                             "Wins0", "Wins1", "Draws", "AvgStep0", "AvgStep1"])
        writer.writerow([map_num, algo_list[0], algo_list[1],
                         np.round(total_reward[0], 2), np.round(total_reward[1], 2),
                         num_win[0], num_win[1], num_win[2],
                         np.round(avg_steps[0], 1), np.round(avg_steps[1], 1)])

# ----------------------------------------------------------------------
# 🚀 Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Olympics-Running Evaluation Script")
    parser.add_argument("--my_ai", default='rl', choices=['rl', 'random'])
    parser.add_argument("--opponent", default='random', choices=['rl', 'random'])
    parser.add_argument("--episode", type=int, default=50)
    parser.add_argument("--map", default='all')
    parser.add_argument("--control_agent", type=int, default=-1)
    parser.add_argument("--control_zone", nargs=4, type=float, default=[50, 70, 20, 40])
    parser.add_argument("--reward_agent", type=int, default=-1)
    parser.add_argument("--reward_bonus", type=float, default=1.0)
    args = parser.parse_args()

    env_type = "olympics-running"
    game = make(env_type, conf=None, seed=1)

    shuffle = False if args.map != 'all' else True
    if not shuffle:
        game.specify_a_map(int(args.map))

    agent_list = [args.opponent, args.my_ai]

    env_control_config = None
    if args.control_agent in [0, 1]:
        env_control_config = {'agent': args.control_agent, 'bounds': args.control_zone}

    reward_bonus_config = None
    if args.reward_agent in [0, 1]:
        reward_bonus_config = {
            'agent': args.reward_agent,
            'bounds': args.control_zone,
            'bonus': args.reward_bonus
        }

    run_game(game, algo_list=agent_list, episode=args.episode,
             shuffle_map=shuffle, map_num=args.map,
             env_control_config=env_control_config,
             reward_bonus_config=reward_bonus_config)







