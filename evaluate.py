import numpy as np


def evaluate(env, agent, num_games):
    achieved_game_points = []
    total_game_steps = []
    total_seen_recipe = []
    total_reached_kitchen = []
    game_name_list = []
    game_max_score_list = []
    game_id = 0

    while True:
        if game_id >= num_games:
            break

        obs, infos = env.reset()
        batch_size = len(obs)

        seen_recipe = [False] * batch_size
        reached_kitchen = [False] * batch_size

        game_name_list += [
            game.metadata["uuid"].split("-")[-1] for game in infos["game"]
        ]
        game_max_score_list += [game.max_score for game in infos["game"]]
        agent.eval()
        agent.init()

        prev_step_dones = [0.0] * batch_size

        prev_h, prev_c = None, None
        still_running_mask = []
        final_scores = []
        print_actions = []

        for _ in range(agent.eval_max_nb_steps_per_episode):
            _, chosen_indices, prev_h, prev_c = agent.act_greedy(
                obs,
                infos['admissible_commands'],
                previous_h=prev_h,
                previous_c=prev_c)
            # send chosen actions to game engine
            chosen_actions_before_parsing = [
                item[idx] for item, idx in zip(infos["admissible_commands"],
                                               chosen_indices)
            ]
            seen_recipe = infos['extra.seen_recipe']
            reached_kitchen = infos['extra.reached_kitchen']

            obs, scores, dones, infos = env.step(chosen_actions_before_parsing)

            still_running = [1.0 - float(item) for item in prev_step_dones]
            prev_step_dones = dones
            final_scores = scores
            still_running_mask.append(still_running)
            print_actions.append(" ".join(chosen_actions_before_parsing[0]
                                          ) if still_running[0] else "--")

            # if all ended, break
            if np.sum(still_running) == 0:
                break

        achieved_game_points += final_scores
        still_running_mask = np.array(still_running_mask)
        total_game_steps += np.sum(still_running_mask, 0).tolist()
        total_seen_recipe += seen_recipe
        total_reached_kitchen += reached_kitchen
        game_id += batch_size

    achieved_game_points = np.array(achieved_game_points, dtype="float32")
    game_max_score_list = np.array(game_max_score_list, dtype="float32")
    normalized_game_points = achieved_game_points / game_max_score_list

    print_strings = []
    print_strings.append(
        "======================================================")
    print_strings.append(game_name_list[0] + ":    " +
                         " | ".join(print_actions))
    print_strings.append(
        "EVAL: rewards: {:2.3f} | normalized reward: {:2.3f} | used steps: {:2.3f} | seen recipe: {:2.3f} | reached kitchen: {:2.3f}"
        .format(np.mean(achieved_game_points), np.mean(normalized_game_points),
                np.mean(total_game_steps), np.mean(total_seen_recipe),
                np.mean(total_reached_kitchen)))
    for i in range(len(game_name_list)):
        print_strings.append(
            "game name: {}, reward: {:2.3f}, normalized reward: {:2.3f}, steps: {:2.3f}, seen recipe: {}, reached kitchen: {}"
            .format(game_name_list[i], achieved_game_points[i],
                    normalized_game_points[i], total_game_steps[i],
                    total_seen_recipe[i], total_reached_kitchen[i]))
    print_strings.append(
        "======================================================")
    print_strings = "\n".join(print_strings)
    print(print_strings)
    return {
        'game_points': np.mean(achieved_game_points),
        'game_points_normalized': np.mean(normalized_game_points),
        'game_steps': np.mean(total_game_steps),
        'seen_recipe': np.mean(total_seen_recipe),
        'reached_kitchen': np.mean(total_reached_kitchen)
    }, print_strings
