# # Unity ML-Agents Toolkit

import sys
sys.path.append('/home/apstwilly/lab/python/ml-agents/ml-agents')
import logging
import argparse
import random
import six
import sys ;  print ( '{0[0]}.{0[1]}' . format ( sys . version_info ) )
import deepmind_lab

from multiprocessing import Process, Queue
import numpy as np
from docopt import docopt

from trainer_controller import TrainerController
from exception import TrainerError


def run_training(sub_id, run_seed, run_options, length, level_script, config):
    """
    Launches training session.
    :param process_queue: Queue used to send signal back to main.
    :param sub_id: Unique id for training session.
    :param run_seed: Random seed used for training.
    :param run_options: Command line arguments for training.
    """
    # Docker Parameters
    docker_target_name = (run_options['docker_target_name']
        if run_options['docker_target_name'] != 'None' else None)

    # General parameters
    env_path = (run_options['env']
        if run_options['env'] != 'None' else None)
    run_id = run_options['run_id']
    load_model = run_options['load']
    train_model = run_options['train']
    save_freq = int(run_options['save_freq'])
    keep_checkpoints = int(run_options['keep_checkpoints'])
    worker_id = int(run_options['worker_id'])
    curriculum_file = (run_options['curriculum']
        if run_options['curriculum'] != 'None' else None)
    lesson = int(run_options['lesson'])
    fast_simulation = not bool(run_options['slow'])
    no_graphics = run_options['no_graphics']
    trainer_config_path = '/home/apstwilly/lab/python/ml-agents/config/trainer_config.yaml'

    # Create controller and launch environment.
    tc = TrainerController(env_path, run_id + '-' + str(sub_id),
                           save_freq, curriculum_file, fast_simulation,
                           load_model, train_model, worker_id + sub_id,
                           keep_checkpoints, lesson, run_seed,
                           docker_target_name, trainer_config_path, no_graphics,
                           length, level_script, config)

    # Begin training
    tc.start_learning()


def main(op):
    try:
        print('''

                        ▄▄▄▓▓▓▓
                   ╓▓▓▓▓▓▓█▓▓▓▓▓
              ,▄▄▄m▀▀▀'  ,▓▓▓▀▓▓▄                           ▓▓▓  ▓▓▌
            ▄▓▓▓▀'      ▄▓▓▀  ▓▓▓      ▄▄     ▄▄ ,▄▄ ▄▄▄▄   ,▄▄ ▄▓▓▌▄ ▄▄▄    ,▄▄
          ▄▓▓▓▀        ▄▓▓▀   ▐▓▓▌     ▓▓▌   ▐▓▓ ▐▓▓▓▀▀▀▓▓▌ ▓▓▓ ▀▓▓▌▀ ^▓▓▌  ╒▓▓▌
        ▄▓▓▓▓▓▄▄▄▄▄▄▄▄▓▓▓      ▓▀      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌   ▐▓▓▄ ▓▓▌
        ▀▓▓▓▓▀▀▀▀▀▀▀▀▀▀▓▓▄     ▓▓      ▓▓▌   ▐▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▌    ▐▓▓▐▓▓
          ^█▓▓▓        ▀▓▓▄   ▐▓▓▌     ▓▓▓▓▄▓▓▓▓ ▐▓▓    ▓▓▓ ▓▓▓  ▓▓▓▄    ▓▓▓▓`
            '▀▓▓▓▄      ^▓▓▓  ▓▓▓       └▀▀▀▀ ▀▀ ^▀▀    `▀▀ `▀▀   '▀▀    ▐▓▓▌
               ▀▀▀▀▓▄▄▄   ▓▓▓▓▓▓,                                      ▓▓▓▓▀
                   `▀█▓▓▓▓▓▓▓▓▓▌
                        ¬`▀▀▀█▓
        ''')
    except:
        print('\n\n\tUnity Technologies\n')

    logger = logging.getLogger('mlagents.trainers')
    '''
    Usage:
      mlagents-learn <trainer-config-path> [options]
      mlagents-learn --help
    Options:
      --env=<file>               Name of the Unity executable [default: None].
      --curriculum=<directory>   Curriculum json directory for environment [default: None].
      --keep-checkpoints=<n>     How many model checkpoints to keep [default: 5].
      --lesson=<n>               Start learning from this lesson [default: 0].
      --load                     Whether to load the model or randomly initialize [default: False].
      --run-id=<path>            The directory name for model and summary statistics [default: ppo].
      --num-runs=<n>             Number of concurrent training sessions [default: 1]. 
      --save-freq=<n>            Frequency at which to save model [default: 50000].
      --seed=<n>                 Random seed used for training [default: -1].
      --slow                     Whether to run the game at training speed [default: False].
      --train                    Whether to train model, or only run inference [default: False].
      --worker-id=<n>            Number to add to communication port (5005) [default: 0].
      --docker-target-name=<dt>  Docker volume to store training-specific files [default: None].
      --no-graphics              Whether to run the environment in no-graphics mode [default: False].
    '''
    options = op

    #logger.info(options)

    num_runs = int(options['num_runs'])
    seed = int(options['seed'])

    if options['env'] == 'None' and num_runs > 1:
        raise TrainerError('It is not possible to launch more than one concurrent training session '
                           'when training from the editor.')

    run_seed = seed
    if seed == -1:
        run_seed = np.random.randint(0, 10000)

    config = {
        'fps': str(options['fps']),
        'width': str(options['width']),
        'height': str(options['height'])
    }
    if options['record']:
        config['record'] = options['record']
    if options['demo']:
        config['demo'] = options['demo']
    if options['demofiles']:
        config['demofiles'] = options['demofiles']
    if options['video']:
        config['video'] = options['video']

    run_training(1, run_seed, options, options['length'], options['level_script'], config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=str, default='/home/apstwilly/lab/python/ml-agents/UnitySDK/Assets/ML-Agents/Examples/3DBall',
                        help='Name of the Unity executable')
    parser.add_argument('-curriculum', type=str, default=None,
                        help=' Curriculum json directory for environment')
    parser.add_argument('--keep-checkpoints', type=int, default=5,
                        help='How many model checkpoints to keep')
    parser.add_argument('--lesson', type=int, default=0,
                        help='Start learning from this lesson')
    parser.add_argument('--load', type=bool, default=False,
                        help='Whether to load the model or randomly initialize')
    parser.add_argument('--run-id', type=str, default='ppo',
                        help='The directory name for model and summary statistics')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of concurrent training sessions')
    parser.add_argument('--save-freq', type=int, default=50000,
                        help='Frequency at which to save model')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed used for training')
    parser.add_argument('--slow', type=bool, default=False,
                        help='Whether to run the game at training speed')
    parser.add_argument('--train', type=bool, default=False,
                        help='Whether to train model, or only run inference')
    parser.add_argument('--worker-id', type=int, default=0,
                        help='Number to add to communication port (5005)')
    parser.add_argument('--docker-target-name', type=str, default=None,
                        help='Docker volume to store training-specific files')
    parser.add_argument('--no-graphics', type=bool, default=False,
                        help='Whether to run the environment in no-graphics mode')

    parser.add_argument('--length', type=int, default=1000,
                        help='Number of steps to run the agent')
    parser.add_argument('--width', type=int, default=80,
                        help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=80,
                        help='Vertical size of the observations')
    parser.add_argument('--fps', type=int, default=60,
                        help='Number of frames per second')
    parser.add_argument('--runfiles_path', type=str, default=None,
                        help='Set the runfiles path to find DeepMind Lab data')
    parser.add_argument('--level_script', type=str,
                        default='tests/empty_room_test',
                        help='The environment level script to load')
    parser.add_argument('--record', type=str, default=None,
                        help='Record the run to a demo file')
    parser.add_argument('--demo', type=str, default=None,
                        help='Play back a recorded demo file')
    parser.add_argument('--demofiles', type=str, default=None,
                        help='Directory for demo files')
    parser.add_argument('--video', type=str, default=None,
                        help='Record the demo run as a video')

    args = parser.parse_args()
    if args.runfiles_path:
        deepmind_lab.set_runfiles_path(args.runfiles_path)
    args = vars(args)
    main(args)
