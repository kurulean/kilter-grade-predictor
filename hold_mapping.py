import re

DB = "data/kilter_data.sqlite"
LAYOUT_ID = 1

CELL = 4
MIN_X, MAX_X = -20, 164
MIN_Y, MAX_Y = 4, 152
GRID_COLS = (MAX_X - MIN_X) // CELL + 1   # 47
GRID_ROWS = (MAX_Y - MIN_Y) // CELL + 1   # 38

FRAME_RE = re.compile(r"p(\d+)r(\d+)")

# product_id=1 roles (see placement_roles table)
START, MIDDLE, FINISH, FOOT = 12, 13, 14, 15
KNOWN_ROLES = {START, MIDDLE, FINISH, FOOT}


def load_placement_lookup(conn, layout_id=LAYOUT_ID):
    """build placement_id -> (x, y, role_id) for one layout's physical holds.

    role_id here is the board's own default role for that position -- a
    climb can and often does override it per-route via the frames string.
    """

    rows = conn.execute(
        """
        SELECT p.id, h.x, h.y, p.default_placement_role_id
        FROM placements p
        JOIN holes h ON p.hole_id = h.id
        WHERE p.layout_id = ?
        """,
        (layout_id,),
    ).fetchall()
    return {pid: (x, y, role) for pid, x, y, role in rows}


def xy_to_cell(x, y):
    """true board (x, y) -> (col, row) on the 47x38 grid.

    raises ValueError if the position falls outside the kept row range
    on purpose, so a hold from beyond y=152 is never silently dropped.
    """
    if not (MIN_X <= x <= MAX_X):
        raise ValueError(f"x={x} outside board extent [{MIN_X}, {MAX_X}]")
    if not (MIN_Y <= y <= MAX_Y):
        raise ValueError(f"y={y} outside the kept grid range [{MIN_Y}, {MAX_Y}]")
    col = (x - MIN_X) // CELL
    row = (y - MIN_Y) // CELL
    return col, row


def frames_to_cells(frames, placement_lookup):
    # parse one climb's frames string into a list of (col, row, role) cells.

    cells = []
    for placement_id_str, role_str in FRAME_RE.findall(frames or ""):
        placement_id = int(placement_id_str)
        role = int(role_str)
        if placement_id not in placement_lookup:
            raise KeyError(
                f"placement_id {placement_id} not found for layout_id={LAYOUT_ID} "
                "check this frames string belongs to the layout you loaded"
            )
        x, y, _default_role = placement_lookup[placement_id]
        col, row = xy_to_cell(x, y)
        cells.append((col, row, role))
    return cells
