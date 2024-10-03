import numpy as np


def kaplan_yorke_dim(lyap_exponents):
    """
    Compute the Kaplan-Yorke dimension from an array of Lyapunov exponents.

    Parameters:
    lyap_exponents (array): Array of Lyapunov exponents sorted in decreasing order.

    Returns:
    kaplan_yorke_dim (float): The Kaplan-Yorke dimension of the system.
    """
    # Calculate the sum of positive Lyapunov exponents
    sum_pos_lyap = np.sum(lyap_exponents[lyap_exponents > 0])

    # Find the index k of the largest Lyapunov exponent that satisfies the condition
    # (sum of positive exponents)/(k+1) <= 0
    k = np.argmax(np.cumsum(lyap_exponents) /
                  (np.arange(len(lyap_exponents))+1) <= 0)

    # Compute the Kaplan-Yorke dimension as k + (sum of positive exponents)/(k+1)
    kaplan_yorke_dim = k + sum_pos_lyap/(k+1)

    return kaplan_yorke_dim
