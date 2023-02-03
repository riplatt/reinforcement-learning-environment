import copy
import datetime
import numpy as np
import os
import string

from PIL import Image, ImageDraw, ImageFont
from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep

BOARD_WIDTH = 4
BOARD_HEIGHT = 4
TILE_IMAGE_PATH = 'resources/E2_Edges.png'

class TileImages:
    def __init__(self):
        self.images = self._crop_and_resize(os.path.join(os.path.dirname(__file__), TILE_IMAGE_PATH))

    def _black_to_transparency(self, img):
        x = np.asarray(img.convert('RGBA')).copy()
        x[:, :, 3] = (255 * (x[:, :, :3] != 0).any(axis=2)).astype(np.uint8)
        return Image.fromarray(x)

    def _crop_and_resize(self, image_path, tile_size=(256, 128), offset=(256, 128)):
        img = Image.open(image_path)
        _images = []
        for j in range(0, 6*128, 128):
            for i in range(0, 4*256, 256):
                cropped_img = img.crop((i, j, i+256, j+128))
                cropped_img = self._black_to_transparency(cropped_img)
                cropped_img = cropped_img.resize((64, 32), Image.LANCZOS)
                _images.append(cropped_img)
        return _images

class Tile:
    def __init__(self, id, sides):
        self.id = id
        self.orientation = 0
        self.sides = sides

    def rotate(self, rotation):
        self.orientation = (self.orientation + 1) % 4
        self.sides = np.roll(self.sides, rotation)

    def render(self):
        side_string = ""
        for side in self.sides:
            side_string += string.ascii_lowercase[side]
        return side_string


class puzzleEnv(py_environment.PyEnvironment):
    def __init__(self, tile_set, discount=0.95):
        super(puzzleEnv, self).__init__(handle_auto_reset=True)

        self._tile_set = tile_set
        self._tile_images = TileImages().images
        self._font_path = os.path.join(os.path.dirname(__file__), 'resources/Roboto-Regular.ttf')

        self.board = self._generate_initial_state(tile_set=tile_set)

        self._number_of_steps = 0
        self._reward = 0
        self._solved_edges = 0
        self._last_action = [-1,0,0]

        # 0 - 1023 action 0-3 & 0x03, tile_1 0-15 >> 2 & 0x0F, tile 0-15 >> 6 & 0x0F
        self._last_action_spec = BoundedArraySpec(
            shape=( ), dtype=np.int32, minimum=0, maximum=1023, name='action')
        self._observation_spec = BoundedArraySpec(
            shape=(64,), dtype=np.int32, minimum=0, maximum=22, name='observation')

        self._discount = np.asarray(discount, dtype=np.float32)
        self._state = 0
        self._board_to_state()
        self._episode_ended = False

    def action_spec(self):
        return self._last_action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        """Return initial_time_step."""
        self._number_of_steps = 0
        self._solved_edges = 0
        self._episode_ended = False
        self.board = self._generate_initial_state(self._tile_set)
        _, _, _ = self._check_state()
        self._reward = 0
        return TimeStep(StepType.FIRST, np.asarray(0.0, dtype=np.float32),
                        self._discount, self._state)

    def _step(self, action):
        """Apply action and return new time_step."""
        action = np.rint(action).astype(np.int32)
        # 1023 = 0011 1111 1111, action[0] = 1023 & 0x03, action[1] = (1023 >> 2) & 0x0F, action[2] = (1023 >> 6) & 0x0F
        _action = [action & 0x03, (action >> 2) & 0x0F, (action >> 6) & 0x0F]

        self._number_of_steps += 1
        is_final, reward, _ = self._check_state()

        # puzzle complete
        if is_final:
            self._reward = reward
            return TimeStep(StepType.LAST, np.asarray(self._reward, dtype=np.float32), self._discount, self._state)

        # out of moves
        if self._episode_ended:
            self._reward = reward
            return TimeStep(StepType.LAST, np.asarray(self._reward, dtype=np.float32), self._discount, self._state)

        # Same Action
        if _action == self._last_action:
            self._reward = 0.0
            return TimeStep(StepType.MID, np.asarray(self._reward, dtype=np.float32), self._discount, self._state)
        self._last_action = _action

        # Illegal Action
        illegal_tile = _action[1] < 0 or _action[1] > 15 or _action[2] < 0 or _action[2] > 15
        illegal_action = _action[0] < 0 or _action[0] > 3
        if illegal_tile or illegal_action:
            self._reward = -0.1
            return TimeStep(StepType.MID, np.asarray(self._reward, dtype=np.float32), self._discount, self._state)

        # do the action
        self._swap_and_rotate(_action[0], _action[1], _action[2])

        is_final, reward, _ = self._check_state()

        step_type = StepType.MID
        if is_final:
            step_type = StepType.LAST

        self._reward = reward
        return TimeStep(step_type, np.asarray(self._reward, dtype=np.float32), self._discount, self._state)

    def _check_state(self):
        _reward_multiplier = 0

        self._board_to_state()

        flat_board = self.board.flatten()
        top_edges = np.asarray([e.sides[0] for e in flat_board]).reshape(4, 4)
        right_edges = np.asarray([e.sides[1] for e in flat_board]).reshape(4, 4)
        bottom_edges = np.asarray([e.sides[2] for e in flat_board]).reshape(4, 4)
        left_edges = np.asarray([e.sides[3] for e in flat_board]).reshape(4, 4)

        top_edges = np.vstack((top_edges, np.zeros((1, top_edges.shape[1]))))
        bottom_edges = np.vstack((np.zeros((1, bottom_edges.shape[1])), bottom_edges))
        right_edges = np.hstack((np.zeros((right_edges.shape[0], 1)), right_edges))
        left_edges = np.hstack((left_edges, np.zeros((left_edges.shape[0], 1))))

        _reward_multiplier += np.count_nonzero(top_edges[0] == 0) / 4 * 10
        _reward_multiplier += np.count_nonzero(bottom_edges[-1] == 0) / 4 * 10
        _reward_multiplier += np.count_nonzero(right_edges[:, [-1]] == 0) / 4 * 10
        _reward_multiplier += np.count_nonzero(left_edges[:, [0]] == 0) / 4 * 10

        # _reward_multiplier += 10 if not top_edges[0].any() else 0
        # _reward_multiplier += 10 if not bottom_edges[-1].any() else 0
        # _reward_multiplier += 10 if not right_edges[:, [-1]].any() else 0
        # _reward_multiplier += 10 if not left_edges[:, [0]].any() else 0

        _reward_multiplier = 1 if _reward_multiplier == 0 else _reward_multiplier

        solved_edges = np.count_nonzero(
            top_edges == bottom_edges) + np.count_nonzero(right_edges == left_edges)

        self._solved_edges = solved_edges

        # out of steps
        if self._number_of_steps > (4 * 4):
            self._episode_ended = True
            reward = -5.0

        # puzzle complete
        is_final = solved_edges >= ((4+1) * 4 + (4+1) * 4)
        if is_final:
            reward = 10.0 * solved_edges * _reward_multiplier
            board_img = self.render(mode='human')
            time_now = datetime.datetime.now()
            image_path = os.path.join(os.path.dirname(__file__), '{}_solved.png'.format(time_now.strftime("%Y%m%d%H%M%S")))
            board_img.save(image_path)
        else:
            # puzzle not complete
            reward = 0.2 * solved_edges * _reward_multiplier

        self._reward = reward
        return is_final, np.asarray(reward, dtype=np.float32), solved_edges

    def _generate_initial_state(self, tile_set, board_size=(4, 4)):
        tiles = np.ndarray((board_size[0]*board_size[1], ), dtype=object)
        for i, tile in enumerate(tile_set):
            tiles[i] = Tile(i, tile)
        # np.random.shuffle(tiles)
        board = tiles.reshape(board_size)
        return board

    def _board_to_state(self):
        _state = np.zeros((4, 4, 4), dtype=np.int32)
        for i in range(4):
            for j in range(4):
                _state[i, j, :] = self.board[i, j].sides
        self._state = _state.flatten()

    def _rotate_tile(self, rotation, board_position):
        tile1_x, tile1_y = np.unravel_index(board_position, self.board.shape)
        self.board[tile1_x, tile1_y].rotate(rotation)

    def _swap_tiles(self, board_position_1, board_position_2):
        tile1_x, tile1_y = np.unravel_index(board_position_1, self.board.shape)
        tile2_x, tile2_y = np.unravel_index(board_position_2, self.board.shape)
        # swap the tiles
        self.board[tile1_x, tile1_y], self.board[tile2_x, tile2_y] = self.board[tile2_x, tile2_y], self.board[tile1_x, tile1_y]

    def _swap_and_rotate(self, rotation, board_position_1, board_position_2):
        self._swap_tiles(board_position_1, board_position_2)
        self._rotate_tile(rotation, board_position_1)

    def get_state(self) -> TimeStep:
        # Returning an unmodifiable copy of the state.
        return copy.deepcopy(self._current_time_step)

    def set_state(self, time_step: TimeStep):
        self._current_time_step = time_step
        self._state = time_step.observation
        # self._board_to_state()

    def render(self, mode='bucas'):
        if mode == 'bucas':
            board_edges = ""
            board_pieces = ""
            for i, tile in enumerate(self.board.flatten()):
                board_pieces += str(tile.id).zfill(3)
                board_edges += tile.render()
            return board_edges, board_pieces
        elif mode == 'human':
            new_img = Image.new('RGBA', (64, 64))
            board = np.empty((4,4), dtype=object)
            for i in range(4):
                for j in range(4):
                    new_img = Image.new('RGBA', (64, 64))
                    tile = self.board[j][i].sides
                    north = self._tile_images[tile[0]]
                    new_img.paste(north, (0, 0), north)
                    east = self._tile_images[tile[1]].rotate(270, expand=True)
                    new_img.paste(east, (32, 0), east)
                    south = self._tile_images[tile[2]].rotate(180, expand=True)
                    new_img.paste(south, (0, 32), south)
                    west = self._tile_images[tile[3]].rotate(90, expand=True)
                    new_img.paste(west, (0, 0), west)
                    board[i][j] = new_img

            # build the board image
            board_img = Image.new('RGBA', (4*64, 4*64))
            for i in range(4):
                for j in range(4):
                    board_img.paste(board[i][j], (i*64, j*64))
            # add stats
            new_img = Image.new('RGB', (265+160, 256))
            new_img.paste(board_img, (160, 0))
            font = ImageFont.truetype(self._font_path, 16)
            draw = ImageDraw.Draw(new_img)
            text = "\nStep: {}\nSolved Edges: {}\nReward: {}\nAction: {}".format(self._number_of_steps, self._solved_edges, self._reward, self._last_action)
            draw.text((0, 0),text,(255,255,255),font=font)
            # draw.textsize(text, font=font)
            return new_img
        else:
            raise ValueError("Invalid render mode: {}".format(mode))
