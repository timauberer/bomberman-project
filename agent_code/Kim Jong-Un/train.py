from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    agent_pos_old = old_game_state("self")[3]
    agent_pos_new = new_game_state("self")[3]

    mean_distance_old = np.mean(np.abs(old_game_state["coins"] - agent_pos_old*np.ones(old_game_state["coins"].size)))
    mean_distance_new = np.mean(np.abs(new_game_state["coins"] - agent_pos_new*np.ones(new_game_state["coins"].size)))
    if(mean_distance_new < mean_distance_old):
        events.append(MEAN_DISTANCE_TO_COINS_DECREASED)

    lowest_distance_to_coin_old = np.min(np.abs(old_game_state["coins"] - agent_pos_old*np.ones(old_game_state["coins"].size)))
    lowest_distance_to_coin_new = np.min(np.abs(new_game_state["coins"] - agent_pos_new*np.ones(new_game_state["coins"].size)))
    if(lowest_distance_to_coin_new < lowest_distance_to_coin_old):
        events.append(LOWEST_DISTANCE_TO_COIN_DECREASED)
    #for i,coin in enumerate(old_game_state["coins"]): 

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.WAITED: -.3
        e.BOMB_DROPPED: .2
        e.CRATE_DESTROYED: .1
        e.COIN_FOUND: .5
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -1.5
        e.GOT_KILLED: -6.5
        MEAN_DISTANCE_TO_COINS_DECREASED: .1
        LOWEST_DISTANCE_TO_COIN_DECREASED: .2
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

"""
def distance(object1: tuple, object2: tuple, field: two_d_array):
    #searching algorithmically shortest walkable path between object1 and object2
    #cursor is the coordinate of the "searching cursor" making its way from object1 to object2
    cursor = object1
    distance = 0
    while(cursor != object2):
        if(np.abs(cursor[0]-object2[0]) > np.abs(cursor[1]-object2[1])):
            if(cursor[0]-object2[0] > 0):
                if(field[cursor[0]+1, cursor[1]] == 0):
                    cursor[0] += 1
                    distance += 1
                else:
            else:
                if(field[cursor[0]-1, cursor[1]] == 0):
                    cursor[0] -= 1
        else:
            if(cursor[1]-object2[1] > 0):
                if(field[cursor[0], cursor[1]+1] == 0):
                    cursor[1] += 1
            else:
                if(field[cursor[0], cursor[1]-1] == 0):
                    cursor[1] -= 1
"""

def distance(object1: tuple, object2: tuple):
    return np.abs(object1[0]-object2[0]) + np.abs(object1[1]-object2[1])
