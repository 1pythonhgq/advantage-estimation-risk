// Apply AFTER the complete task/model PPO config (e.g. rho1bSft2 PPO GSM8K).
// This overlay preserves its resolved optimizer, reward, GAE and batch settings.
// It does not select diagnostic-safe training questions or a training budget.
// The actor engine is retained for compatibility, but never backward/stepped.
{
  trainer+: {
    disable_actor_training: true,
    disable_critic_training: false,
    enable_exponential_moving_average_actor: false,
    save_hf_critic_checkpoint: true,
  },
}
