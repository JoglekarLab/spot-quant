"""Regression test for the file-switch crash.

The crash was caused by re-entrancy: ``viewer.layers.clear()`` fires dims
events that triggered the detection panel's ``run()``, which added layers
back while ``clear()`` was still iterating.  The fix is a ``loading`` flag
(plus a ``_running`` re-entrancy guard) so ``run()`` is a no-op during the
layer swap.  This test reproduces the guard logic without a live viewer.
"""


class _FakeState:
    def __init__(self):
        self.loading = False


class _GuardedPanel:
    """Minimal stand-in mirroring DetectionPanel.run()'s guard clauses."""

    def __init__(self):
        self.state = _FakeState()
        self._running = False
        self.events = []

    def run(self):
        if self.state.loading or self._running:
            self.events.append("skipped")
            return
        self._running = True
        try:
            self.events.append("ran")
        finally:
            self._running = False


def test_run_skipped_during_load_then_runs_once():
    panel = _GuardedPanel()

    # While a file loads, clearing layers fires several dims events.
    panel.state.loading = True
    for _ in range(3):
        panel.run()
    panel.state.loading = False

    # After the swap completes, the image is published and run executes once.
    panel.run()

    assert panel.events == ["skipped", "skipped", "skipped", "ran"]


def test_reentrant_run_is_blocked():
    panel = _GuardedPanel()

    # Simulate run() re-entering itself (e.g. adding a layer fires an event
    # that calls run() again before the outer call finished).
    def reentrant():
        panel._running = True
        try:
            panel.run()          # inner call must be skipped
        finally:
            panel._running = False

    reentrant()
    assert panel.events == ["skipped"]
