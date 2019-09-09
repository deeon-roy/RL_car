from flat_game import carmunk
import numpy as np
from nn import neural_net
from learning import rotate_state, flatten_state
NUM_INPUTS = 12


def play(model):

    car_distance = 0
    game_state = carmunk.GameState()
    four_states = [[None]*3]*4

    # Do nothing to get initial.
    _, start_state = game_state.frame_step((2))

    rotate_state(four_states, start_state) 

    # Move.
    while True:
        car_distance += 1
        flat_four = np.array([flatten_state(four_states)])

        # Choose action.
        action = (np.argmax(model.predict(flat_four, batch_size=1)))

        # Take action.
        _, state = game_state.frame_step(action)

        rotate_state(four_states, state) 
        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)


if __name__ == "__main__":
    saved_model = 'saved-models/128-128-64-50000-25000.h5'
    model = neural_net(NUM_INPUTS, [128, 128], saved_model)
    play(model)
