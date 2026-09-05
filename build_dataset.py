import sqlite3 
import numpy as np
import hold_mapping as hm

from hold_mapping import load_placement_lookup, frames_to_cells

# constants
DB = "data/kilter_data.sqlite"
LAYOUT_ID = 1
MIN_ASCENTS_TRAIN = 2
MIN_ASCENTS_TEST = 3
ROLE_TO_CHANNEL = {12: 0, 13: 1, 14: 2, 15: 3} # start, middle, finish, foot
NUM_CHANNELS = 4
MAX_GRADE = 13 # drop v14, too few samples to learn from
MAX_ANGLE = 70.0 # highest board angle, used to scale angle to 0 to 1

# turn a resolved hold list into one grid
def build_image(cells):
    img = np.zeros((NUM_CHANNELS, hm.GRID_ROWS, hm.GRID_COLS), dtype = np.uint8)
    for col, row, role in cells:
        # a handful of climbs use role ids from other products (not 12-15)
        # skip those holds, there's no channel defined for them
        channel = ROLE_TO_CHANNEL.get(role)
        if channel is not None:
            img[channel, row, col] = 1
    return img

# turn grade string into integer
def parse_grade(boulder_grade):
    return int(boulder_grade.split("/")[-1].replace("V", ""))

# loop once per split, a fresh connection and lookup each time since layout_id
# is the same for all three, this just keeps each split's query self-contained
images, angles, labels, splits = [], [], [], []
for split_name, table in [("train", "kilter_train"), ("val", "kilter_val"), ("test", "kilter_test")]:
    conn = sqlite3.connect(DB)
    lookup = load_placement_lookup(conn, layout_id=LAYOUT_ID)
    query = "SELECT frames, angle, boulder_grade, ascensionist_count FROM {} WHERE layout_id=?".format(table)
    for frames, angle, grade, ascents in conn.execute(query, (LAYOUT_ID,)):
        if split_name == "train" and ascents <= MIN_ASCENTS_TRAIN:
            continue
        if split_name == "test" and ascents <= MIN_ASCENTS_TEST:
            continue
        grade_num = parse_grade(grade)
        if grade_num > MAX_GRADE:
            continue
        cells = frames_to_cells(frames, lookup)
        images.append(build_image(cells))
        angles.append(angle / MAX_ANGLE)
        labels.append(grade_num)
        splits.append(split_name)
    conn.close()


# images is mostly zeros, so this compresses the data and saves disk memory
np.savez_compressed(
    "data/kilter_images.npz",
    images = np.stack(images),
    angles = np.array(angles, dtype = np.float32),
    labels = np.array(labels, dtype = np.int64),
    split = np.array(splits),
)
print(f"saved {len(images)} climbs, image shape {images[0].shape}")
