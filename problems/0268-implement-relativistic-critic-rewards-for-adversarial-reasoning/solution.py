def compute_raro_rewards(
    critic_prediction: str,
    expert_position: int,
    tau_critic: float = 0.5,
    tau_policy: float = 0.5
) -> tuple[float, float]:
    """
    Compute rewards for the critic and policy in the RARO adversarial game.
    
    In RARO, a relativistic critic compares two answers (one from expert, one from policy)
    and predicts which is better. The critic and policy receive rewards based on whether
    the critic correctly identifies the expert answer.
    
    Args:
        critic_prediction: The critic's prediction - one of 'expert', 'policy', or 'tie'
        expert_position: Which position (1 or 2) contains the expert answer in the pair
                        (the other position contains the policy answer)
        tau_critic: Reward given to critic when it predicts 'tie' (default: 0.5)
        tau_policy: Reward given to policy when critic predicts 'tie' (default: 0.5)
    
    Returns:
        Tuple of (critic_reward, policy_reward) where:
        - 