import gymnasium as gym
import numpy as np
from typing import Optional

from deep_learning_rl_sm.environments.our_gym import OurEnv


def _get_info():
    # don't think we need this
    return 0


class ConnectFour(OurEnv):

    def __init__(self, width: int = 7, length: int = 6):
        # The size of the square grid
        self.width = width
        self.length = length

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._curr_state = np.zeros((self.length, self.width))

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.MultiDiscrete(np.full((self.length, self.width), 3)),
            }
        )
        self.action_mask = np.array([True for _ in range(7)], dtype=np.int8)
        self.no_actions = 7
        self.action_space = gym.spaces.Discrete(self.no_actions)
        self.action_dim = 1
        self.state_dim = np.shape(self._curr_state)

        self.max_two_p_game_length = 21

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._curr_state = np.zeros((self.length, self.width))
        self.action_mask = np.array([True for _ in range(7)], dtype=np.int8)
        obs = self._curr_state
        inf = _get_info()  # info not used yet (not sure if we need this tbh)

        return obs, inf

    def check_win(self, player):
        # Check horizontal lines
        for row in self._curr_state:
            for col in range(len(row) - 3):
                if row[col] == player and row[col + 1] == player and row[col + 2] == player and row[col + 3] == player:
                    return True

        # Check vertical lines
        for col in range(len(self._curr_state[0])):
            for row in range(len(self._curr_state) - 3):
                if self._curr_state[row][col] == player and self._curr_state[row + 1][col] == player and \
                        self._curr_state[row + 2][col] == player and self._curr_state[row + 3][col] == player:
                    return True

        # Check diagonal lines (downward slope)
        for row in range(len(self._curr_state) - 3):
            for col in range(len(self._curr_state[0]) - 3):
                if self._curr_state[row][col] == player and self._curr_state[row + 1][col + 1] == player and \
                        self._curr_state[row + 2][
                            col + 2] == player and self._curr_state[row + 3][col + 3] == player:
                    return True

        # Check diagonal lines (upward slope)
        for row in range(3, len(self._curr_state)):
            for col in range(len(self._curr_state[0]) - 3):
                if self._curr_state[row][col] == player and self._curr_state[row - 1][col + 1] == player and \
                        self._curr_state[row - 2][col + 2] == player and self._curr_state[row - 3][col + 3] == player:
                    return True

        return False

    def board_full(self):
        return np.all(self._curr_state[0, :] != 0)

    def step(self, action):
        # player move:
        # iff top row full for selected column throw exception
        if self._curr_state[0, action] != 0:
            print("invalid actions should never be taken. (i.e. this column is full)...")
            print()
            print(self._curr_state)
            print()
            print(action)
            raise Exception
        insert_row = np.where(self._curr_state[:, action] == 0)[0][-1]
        self._curr_state[insert_row, action] = 1  # player indicates its pieces by a 1

        # check win condition
        player_winner = False
        adv_winner = False
        if self.check_win(1):
            player_winner = True

        if not player_winner:
            # adv move
            mask = np.array([1 if self._curr_state[0, i] == 0 else 0 for i in range(self.no_actions)], dtype=np.int8)
            adv_action = self.action_space.sample(mask=mask)
            adv_insert_row = np.where(self._curr_state[:, adv_action] == 0)[0][-1]
            self._curr_state[adv_insert_row, adv_action] = 2  # player indicates its pieces by a 2

            # check win condition
            if self.check_win(2):
                adv_winner = True

        # check board full
        is_full = self.board_full()

        term = is_full or player_winner or adv_winner
        trunc = False  # don't think we need this for this game
        # (truncated is a condition for early stopping without a terminal state being reached)
        rew = 1 if player_winner else -1 if adv_winner else 0
        obs = self._curr_state
        inf = _get_info()

        # recalculate action mask
        for idx, top_row_entry in enumerate(self._curr_state[0]):
            if top_row_entry != 0:
                self.action_mask[idx] = False
        return obs, rew, term, trunc, inf

    def step_2P(self, action, player):
        # player move:
        # iff top row full for selected column throw exception
        if self._curr_state[0, action] != 0:
            print("invalid actions should never be taken. (i.e. this column is full)...")
            print()
            print(self._curr_state)
            print()
            print(action)
            raise Exception
        insert_row = np.where(self._curr_state[:, action] == 0)[0][-1]
        self._curr_state[insert_row, action] = player  # player indicates its pieces by a 1

        # recalculate action mask
        for idx, top_row_entry in enumerate(self._curr_state[0]):
            if top_row_entry != 0:
                self.action_mask[idx] = False

        # check win condition
        player_winner = False
        if self.check_win(player):
            player_winner = True

        # check board full
        is_full = self.board_full()

        term = is_full or player_winner
        trunc = False  # don't think we need this for this game
        # truncated is a condition for early stopping without a terminal state being reached
        opposite_player = 1 if player == 2 else 2  # Current player = 1 / 2 --> Opposite player = 2 / 1
        rew = 1 if player_winner else -1 if self.check_win_potential(opposite_player) else 0
        obs = self._curr_state
        inf = _get_info()

        return obs, rew, term, trunc, inf

    def display_board(self):
        print(self._curr_state)

    def check_win_potential(self, player):
        row_pos_potential_moves = []

        # find legal moves
        for i in range(self.width):
            empty_rows = np.where(self._curr_state[:, i] == 0)[0]
            if 0 < len(empty_rows):
                row_pos_potential_moves += [empty_rows[-1]]
            else:
                row_pos_potential_moves += [None]
        # check if any leads to win for given player
        for i, mask_entry in enumerate(self.action_mask):
            if mask_entry != 1:
                continue
            self._curr_state[row_pos_potential_moves[i]][i] = player
            if self.check_win(player=player):
                self._curr_state[row_pos_potential_moves[i]][i] = 0
                return True
            self._curr_state[row_pos_potential_moves[i]][i] = 0

        return False


"""game = ConnectFour()
game.reset()
game._curr_state = np.array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 1.], [0., 0., 2., 0., 0., 0., 2.],
                    [2., 1., 1., 1., 2., 0., 1.], [2., 1., 1., 2., 2., 0., 2.], [1., 2., 1., 2., 2., 0., 1.]])
game.step_2P(6, 2)
game.display_board()
print()
print(game.check_win_potential(1))
print()
game.display_board()"""

