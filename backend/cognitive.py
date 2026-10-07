def cognitive_load(eye, head, silence):
    load = (eye * 2) + (head * 1.5)
    if silence:
        load += 0.3
    return min(load, 1.0)