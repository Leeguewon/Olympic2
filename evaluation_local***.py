# 결과값  이미지: result2.png
# ..................................................
# total reward:  [139.41847923  62.        ]
# Result in map all within 50 episode:
# +-----------+--------+------+
# |   Name    | random |  rl  |
# +-----------+--------+------+
# |   score   | 139.42 | 62.0 |
# |    win    |   0    |  31  |
# | avg_steps | 131.1  |  -   |
# +-----------+--------+------+


# import numpy as np
# import random
# import argparse
# import os
# from tabulate import tabulate
# import sys
# import csv

# # ----------------------------------------------------------------------
# # ⚙️ Dummy 설정 (환경/에이전트 없는 로컬 테스트용)
# # ----------------------------------------------------------------------

# class DummyRLAgent:
#     """rl_agent가 없을 때를 대비한 더미 클래스. 랜덤 행동을 반환."""
#     def choose_action(self, obs):
#         return random.randint(0, 35)

# try:
#     from agents.rl.submission import agent as rl_agent
# except ImportError:
#     rl_agent = DummyRLAgent()

# def make(env_type, conf=None, seed=1):
#     print(f"\n❌ FATAL ERROR: Environment '{env_type}' not found.")
#     print("Check 'env/chooseenv.py' or your project structure.")
#     sys.exit(1)

# try:
#     from env.chooseenv import make
# except ImportError:
#     pass

# # ----------------------------------------------------------------------
# # 🧭 Action Map (힘 -100~200, 각도 -30~30)
# # ----------------------------------------------------------------------
# ACTION_MAP = {
#     0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30],
#     6: [-40, -30], 7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30],
#     12: [20, -30], 13: [20, -18], 14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30],
#     18: [80, -30], 19: [80, -18], 20: [80, -6], 21: [80, 6], 22: [80, 18], 23: [80, 30],
#     24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6], 28: [140, 18], 29: [140, 30],
#     30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18], 35: [200, 30]
# }

# RENDER = False

# # ----------------------------------------------------------------------
# # 🎮 행동 선택
# # ----------------------------------------------------------------------
# def get_joint_actions(state, algo_list):
#     joint_actions = []
#     for agent_idx, algo in enumerate(algo_list):
#         if algo == 'random':
#             force = float(random.uniform(-100, 200))
#             angle = float(random.uniform(-30, 30))
#         elif algo == 'rl':
#             try:
#                 obs = state[agent_idx]['obs'].flatten()
#                 actions_raw = rl_agent.choose_action(obs)
#                 if isinstance(actions_raw, (int, np.integer)):
#                     force, angle = ACTION_MAP[actions_raw]
#                 else:
#                     force, angle = actions_raw[0], actions_raw[1]
#             except Exception:
#                 force, angle = 0, 0
#         else:
#             force, angle = 0, 0

#         force = float(np.clip(force, -100, 200))
#         angle = float(np.clip(angle, -30, 30))
#         joint_actions.append([[force], [angle]])
#     return joint_actions


# # ----------------------------------------------------------------------
# # 💥 보상 함수 (기존 유지)
# # ----------------------------------------------------------------------
# def smoothness_reward(prev_action, curr_action):
#     if prev_action is None or curr_action is None:
#         return 0.0
#     d_angle = abs(curr_action[1] - prev_action[1])
#     d_force = abs(curr_action[0] - prev_action[0])
#     smooth = 1.0 - 0.015 * d_angle - 0.002 * d_force
#     return max(smooth, 0.0)

# def collision_penalty(info):
#     if isinstance(info, dict) and info.get("collision", False):
#         return -10.0
#     return 0.0


# # ----------------------------------------------------------------------
# # 🏁 메인 실행 루프 (3번째 코드 스타일)
# # ----------------------------------------------------------------------
# def run_game(env, algo_list, episode, shuffle_map, map_num,
#              env_control_config=None, reward_bonus_config=None, verbose=False):

#     total_reward = np.zeros(2, dtype=float)
#     num_win = np.zeros(3, dtype=int)
#     total_steps = []  # ✅ 통합 걸음수 리스트
#     prev_action = None
#     episode = int(episode)

#     for i in range(1, episode + 1):
#         episode_reward = np.zeros(2, dtype=float)
#         state = env.reset(shuffle_map)
#         if RENDER:
#             env.env_core.render()

#         step = 0
#         while True:
#             if step < 1:
#                 joint_action = [[[150.0], [0.0]], [[150.0], [0.0]]]
#             else:
#                 joint_action = get_joint_actions(state, algo_list)

#             try:
#                 next_state, reward, done, _, info = env.step(joint_action)
#             except Exception:
#                 done = True
#                 reward = [0.0, 0.0]
#                 info = {}

#             reward = np.array(reward, dtype=float)
#             a0 = (joint_action[0][0][0], joint_action[0][1][0])
#             reward[0] += smoothness_reward(prev_action, a0)
#             reward[0] += collision_penalty(info)
#             prev_action = a0

#             episode_reward += reward
#             step += 1

#             if done:
#                 # ✅ 승리한 경기의 step만 저장 (누가 이기든)
#                 if reward[0] != reward[1]:
#                     if reward[0] == 100:
#                         num_win[0] += 1
#                         total_steps.append(step)
#                     elif reward[1] == 100:
#                         num_win[1] += 1
#                         total_steps.append(step)
#                 else:
#                     num_win[2] += 1

#                 if not verbose:
#                     print('.', end='')
#                     if i % 50 == 0 or i == episode:
#                         print()
#                 break

#             state = next_state

#         total_reward += episode_reward

#     # 평균 계산
#     total_reward /= episode
#     average_steps = np.mean(total_steps) if total_steps else 0

#     # ------------------------------------------------------------------
#     # 🧾 결과 출력 (3번째 코드 포맷 동일)
#     # ------------------------------------------------------------------
#     print("total reward: ", total_reward)
#     print(f"Result in map {map_num} within {episode} episode:")
#     header = ['Name', algo_list[0], algo_list[1]]
#     data = [
#         ['score', np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
#         ['win', num_win[0], num_win[1]],
#         ['avg_steps', np.round(average_steps, 1), '-']
#     ]
#     print(tabulate(data, headers=header, tablefmt='pretty'))

#     # ------------------------------------------------------------------
#     # 💾 CSV 기록 (선택 사항)
#     # ------------------------------------------------------------------
#     result_path = "results.csv"
#     file_exists = os.path.exists(result_path)
#     with open(result_path, "a", newline="") as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(["Map", "Agent0", "Agent1", "Score0", "Score1",
#                              "Wins0", "Wins1", "Draws", "AvgSteps"])
#         writer.writerow([map_num, algo_list[0], algo_list[1],
#                          np.round(total_reward[0], 2), np.round(total_reward[1], 2),
#                          num_win[0], num_win[1], num_win[2],
#                          np.round(average_steps, 1)])


# # ----------------------------------------------------------------------
# # 🚀 Main
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Olympics-Running Evaluation Script")
#     parser.add_argument("--my_ai", default='rl', choices=['rl', 'random'])
#     parser.add_argument("--opponent", default='random', choices=['rl', 'random'])
#     parser.add_argument("--episode", type=int, default=50)
#     parser.add_argument("--map", default='all')
#     args = parser.parse_args()

#     env_type = "olympics-running"
#     game = make(env_type, conf=None, seed=1)

#     shuffle = False if args.map != 'all' else True
#     if not shuffle:
#         game.specify_a_map(int(args.map))

#     agent_list = [args.opponent, args.my_ai]
#     run_game(game, algo_list=agent_list, episode=args.episode,
#              shuffle_map=shuffle, map_num=args.map, verbose=False)