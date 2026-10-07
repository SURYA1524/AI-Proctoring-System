def detect_emotion(eye, head):
    if eye < 0.02 and head < 0.05:
        return "Calm"
    elif eye < 0.05:
        return "Focused"
    else:
        return "Stressed"