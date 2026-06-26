"""Lightweight regression test for tactile GT recording order (no Isaac Sim)."""

from types import SimpleNamespace

import torch


class _TactileEvalStub:
  action_horizon = 4
  replan_every = 4
  eval_predict_tactile = True
  _exec_count = 0
  _action_buffer = []
  _tactile_eval_active = False
  _tactile_chunk_start = 0
  _pred_tactiles = None
  _gt_tac_left = []
  _gt_tac_right = []
  _current_task_save_root = None
  finalized = 0

  def _extract_tactile_pair(self, observation):
    step = observation["step"]
    tensor = torch.full((3, 8, 8), float(step))
    return tensor, tensor + 1

  def _start_tactile_eval_chunk(self, task, pred_tactiles):
    self._tactile_eval_active = True
    self._tactile_chunk_start = self._exec_count
    self._pred_tactiles = pred_tactiles
    self._gt_tac_left = []
    self._gt_tac_right = []
    self._current_task_save_root = task.save_root

  def _record_tactile_gt(self, observation):
    if not self._tactile_eval_active:
      return
    idx = self._exec_count - self._tactile_chunk_start
    if idx < 0 or idx >= self.action_horizon:
      return
    left_tac, right_tac = self._extract_tactile_pair(observation)
    if len(self._gt_tac_left) <= idx:
      self._gt_tac_left.extend([None] * (idx + 1 - len(self._gt_tac_left)))
      self._gt_tac_right.extend([None] * (idx + 1 - len(self._gt_tac_right)))
    self._gt_tac_left[idx] = left_tac
    self._gt_tac_right[idx] = right_tac

  def _finalize_tactile_eval_chunk(self, task):
    assert self._gt_tac_left[0] is not None, "chunk index 0 must be recorded"
    assert all(item is not None for item in self._gt_tac_left[: self.action_horizon])
    self.finalized += 1
    self._tactile_eval_active = False

  def eval_step(self, observation):
    need_replan = len(self._action_buffer) == 0 or (
      self.replan_every > 0
      and self._exec_count > 0
      and self._exec_count % self.replan_every == 0
    )
    if need_replan:
      if self._tactile_eval_active:
        self._finalize_tactile_eval_chunk(SimpleNamespace(save_root="."))
      self._action_buffer = list(range(self.action_horizon))
      self._start_tactile_eval_chunk(SimpleNamespace(save_root="."), pred_tactiles=[0])
    if self.eval_predict_tactile:
      self._record_tactile_gt(observation)
    self._action_buffer.pop(0)
    self._exec_count += 1


def main():
  stub = _TactileEvalStub()
  task = SimpleNamespace(save_root=".")
  for step in range(9):
    stub.eval_step({"step": step})
  assert stub.finalized == 2, f"expected 2 finalized chunks, got {stub.finalized}"
  print("tactile_eval_record_test: OK")


if __name__ == "__main__":
  main()
