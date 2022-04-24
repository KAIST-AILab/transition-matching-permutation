import json

from agent import Agent
import evaluate
import generic
import reinforcement_learning_dataset


def run_eval():

    config = generic.load_config()
    agent = Agent(config)
    output_dir = f"checkpoints/{agent.experiment_tag}"

    # make game environments
    requested_infos = agent.select_additional_infos()
    games_dir = "./"

    eval_env, num_eval_game = reinforcement_learning_dataset.get_evaluation_game_env(
        games_dir + config['rl']['data_path'],
        config['rl']['difficulty_level'],
        requested_infos,
        agent.eval_max_nb_steps_per_episode,
        agent.eval_batch_size,
        config,
        valid_or_test="test")

    # load pretrained models
    agent.load_pretrained_model(agent.load_from_tag + ".pt",
                                load_partial_graph=False)

    # evaluate
    eval_stats, detailed_scores = evaluate.evaluate(eval_env, agent,
                                                      num_eval_game)
    eval_stats = {k: str(v) for k, v in eval_stats.items()}

    # write accuracies down into file
    _s = json.dumps({**eval_stats, "detailed scores": detailed_scores})
    with open(output_dir + "/log.json", 'a+') as outfile:
        outfile.write(_s + '\n')
        outfile.flush()


if __name__ == '__main__':
    run_eval()
