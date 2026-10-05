"""Raised inside a mutation trial and translated by the plugin into SKIPPED or ERROR statuses."""


class TrialSkipped(Exception):
    """The test called pytest.skip during a trial."""


class TrialError(Exception):
    """A trial could not produce a pass/fail outcome; the message is the recorded reason."""
