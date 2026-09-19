"""Global stage dimensions. Set them before constructing a problem."""

N_X = 24  # interior spatial nodes (state dimension per stage)
N_U = 24  # control dimension per stage (== N_X for distributed control B=I)


def set_dims(nx: int, nu: int = None) -> None:
    """Set state and control dimensions; controls default to the state dimension."""
    global N_X, N_U
    N_X = int(nx)
    N_U = int(nx if nu is None else nu)
