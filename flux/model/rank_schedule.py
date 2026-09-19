def get_rank_by_timestep(
    timestep,
    max_timestep,
    max_rank,
    min_rank=1,
    rank_schedule="decreasing",
):
    """Return the effective adapter rank for a diffusion timestep."""
    timestep = float(timestep)
    if rank_schedule == "increasing":
        progress = timestep / max_timestep
    elif rank_schedule == "decreasing":
        progress = (max_timestep - timestep) / max_timestep
    else:
        raise ValueError("Unsupported rank schedule: {!r}".format(rank_schedule))

    return int(progress * (max_rank - min_rank)) + min_rank
