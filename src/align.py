import numpy as np 


def prob_sw(pwm, seqs, w=.05):
    """
    Args:
        pwm: numpy tensor representing a position-specific scoring matrix of
            per-sequence, per-position probabilities. Has shape BxKxM.
        seqs: one-hot encoded np tensor of shape KxL.

    Returns:
        float: The score for the highest scoring local alignment.
        tuple(int, int): The index of the highest scoring local alignment.
    """
    B, K, M, L = seqs.shape[0], pwm.shape[0], pwm.shape[1], seqs.shape[2]
    # pwm = pwm[np.newaxis, :, :].repeat(B, axis=0)
    A = np.matmul(pwm.T, seqs)  # shape: MxL
    print(A.shape)
    SM = np.zeros((B, M + 1, L + 1))
    SM[:, 1:, 1:] = A
    for i in range(M):
        for j in range(L):
            SM[:, i + 1, j + 1] = np.max(np.stack(
                (np.zeros(B),  # new local alignment
                 A[:, i, j] + SM[:, i, j],  # match/substitute
                 SM[:, i + 1, j] - w,  # gap: filter
                 SM[:, i, j + 1] - w),  # gap: sequence
                axis=1
            ), axis=1)
    SM = SM[:, 1:, 1:]
    max_score = np.max(np.max(SM, axis=1), axis=1)
    end_idx = np.argmax(np.max(SM, axis=1), axis=1)
    return max_score, end_idx
