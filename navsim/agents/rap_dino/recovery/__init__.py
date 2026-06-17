__all__ = ["make_recovery_trajectory"]


def __getattr__(name):
    if name == "make_recovery_trajectory":
        from .recovery_target import make_recovery_trajectory

        return make_recovery_trajectory
    raise AttributeError(name)
