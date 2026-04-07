"""
Mode definitions for trajectory prediction.
Defines both fine-grained modes (turn + speed) and turn-only modes.
"""

TURN_MODES = ['TurnLeft', 'TurnRight', 'Straight', 'Hold']
SPEED_MODES = ['Accel', 'Decel', 'Normal', 'Hold']

NUM_TURN = len(TURN_MODES)
NUM_SPEED = len(SPEED_MODES)

MODE_MAP = {
    f"{t}_{s}": i
    for i,(t,s) in enumerate(
        [(t,s) for t in TURN_MODES for s in SPEED_MODES]
    )
}

# Turn-only mode mapping (4 modes)
TURN_MODE_MAP = {
    mode: i for i, mode in enumerate(TURN_MODES)
}

MODE_NAMES = {v:k for k,v in MODE_MAP.items()}

VALID_MODES = [
    MODE_MAP["TurnLeft_Accel"],
    MODE_MAP["TurnLeft_Decel"],
    MODE_MAP["TurnLeft_Normal"],
    MODE_MAP["TurnRight_Accel"],
    MODE_MAP["TurnRight_Decel"],
    MODE_MAP["TurnRight_Normal"],
    MODE_MAP["Straight_Accel"],
    MODE_MAP["Straight_Decel"],
    MODE_MAP["Straight_Normal"],
    MODE_MAP["Hold_Hold"],
]

# Mapping from fine-grained mode indices to turn mode indices
FINE_TO_TURN_MAP = {}
for fine_mode, idx in MODE_MAP.items():
    turn_name = fine_mode.split('_')[0]  # Extract turn part (e.g., "TurnLeft")
    FINE_TO_TURN_MAP[idx] = TURN_MODE_MAP[turn_name]

# Turn mode names for visualization and logging
TURN_MODES_NAMES = {i: name for i, name in enumerate(TURN_MODES)}

# Valid turn modes (all 4 are valid)
VALID_TURN_MODES = [0, 1, 2, 3]