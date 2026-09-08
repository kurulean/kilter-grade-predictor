export type HoldType = "normal" | "start" | "finish" | "foot";

export type Hold = {
  id: number;
  x: number;
  y: number;
  width: number;
  height: number;
  cx: number;
  cy: number;
  area: number;
  path: string;
};

export type SelectedHold = {
  id: number;
  type: HoldType;
};

export type Climb = {
  name: string;
  grade: string;
  holds: SelectedHold[];
};
