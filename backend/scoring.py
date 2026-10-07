def compute_score(eye, head, base):
    eye_diff = abs(eye - base["eye"])
    head_diff = abs(head - base["head"])

    score = (eye_diff * 15) + (head_diff * 10)
    return min(score, 1.0)