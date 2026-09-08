// Real board geometry, matching hold_mapping.py exactly.
export const GRID_COLS = 47;
export const GRID_ROWS = 38;
export const CELL = 15;
export const LEFT = 34;
export const TOP = 20;

export function colX(col: number): number {
  return LEFT + col * CELL + CELL / 2;
}

export function rowY(row: number): number {
  return TOP + (GRID_ROWS - 1 - row) * CELL + CELL / 2;
}

export const VIEW_W = LEFT + GRID_COLS * CELL + 14;
export const VIEW_H = TOP + GRID_ROWS * CELL + 10;

export type Role = "start" | "middle" | "finish" | "foot";

export const ROLES: Role[] = ["start", "middle", "finish", "foot"];

export const ROLE_TO_CODE: Record<Role, number> = {
  start: 12,
  middle: 13,
  finish: 14,
  foot: 15,
};

export const CODE_TO_ROLE: Record<number, Role> = {
  12: "start",
  13: "middle",
  14: "finish",
  15: "foot",
};

export const ROLE_TO_CHANNEL: Record<Role, number> = {
  start: 0,
  middle: 1,
  finish: 2,
  foot: 3,
};

export type HoldKey = string; // "col,row"
export const holdKey = (col: number, row: number): HoldKey => `${col},${row}`;
