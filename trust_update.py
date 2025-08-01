import numpy as np

def update_trust(data, labels, pattern, presence, permanence, logic, iterations=10, alpha=0.1, beta=0.4, gamma=0.1, delta=0.2, dataset_name=""):
    S_t = 0.5 * np.ones(len(data))
    T_t = 0.5 * np.ones(len(data))
    history = {'S': [], 'T': [], 'accuracy': [], 'mean_V_q': [], 'mean_V_b': [], 'mean_V_l': []}
    preds, probs, acc = pattern.validate(data, labels)
    for t in range(iterations):
        V_q = presence.validate(probs)
        V_b = permanence.validate(preds, probs)
        V_l = logic.validate(data, preds)
        V_t = beta * V_b + (1 - beta - delta) * V_q + delta * V_l
        S_t = S_t + gamma * (V_t - S_t)
        T_t = alpha * V_t + (1 - alpha) * T_t
        weighted_probs = probs * V_q[:, None] * V_b[:, None] * V_l[:, None]
        preds = np.argmax(weighted_probs, axis=1)
        acc = (preds == labels).mean()
        history['S'].append(np.mean(S_t))
        history['T'].append(np.mean(T_t))
        history['accuracy'].append(acc)
        history['mean_V_q'].append(np.mean(V_q))
        history['mean_V_b'].append(np.mean(V_b))
        history['mean_V_l'].append(np.mean(V_l))
    return history