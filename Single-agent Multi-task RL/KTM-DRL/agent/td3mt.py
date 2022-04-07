import torch
import torch.nn.functional as F

from agent.td3 import TD3


class TD3MT(TD3):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 num_env,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 cuda_index=None
                 ):

        super().__init__(state_dim, action_dim, max_action,
                         discount, tau,
                         policy_noise, noise_clip,
                         policy_freq, cuda_index)
        self.it = 0
        self.total_it = [0 for _ in range(num_env)]
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor_optimizer_online = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer_online = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def save(self, filename):
        super().save(filename)

        torch.save(self.actor_optimizer_online.state_dict(), filename + "_actor_optimizer_online.pt")
        torch.save(self.critic_optimizer_online.state_dict(), filename + "_critic_optimizer_online.pt")

    def load(self, filename):
        super().load(filename)

        self.actor_optimizer_online.load_state_dict(torch.load(filename + "_actor_optimizer_online.pt"))
        self.critic_optimizer_online.load_state_dict(torch.load(filename + "_critic_optimizer_online.pt"))

    def pad_state(self, state):
        return torch.cat([state,
                          torch.zeros(state.shape[0], self.state_dim - state.shape[1]).to(self.device)],
                         dim=1)

    def pad_action(self, action):
        return torch.cat([action,
                          torch.zeros(action.shape[0], self.action_dim - action.shape[1]).to(self.device)],
                         dim=1)

    def train_mt(self, idx, teacher, replay, batch_size=100, is_offline=True):
        self.total_it[idx] += 1

        state, action, next_state, reward, not_done = replay.sample(batch_size)

        state_dim_org = state.shape[1]
        action_dim_org = action.shape[1]

        with torch.no_grad():

            state_pad = self.pad_state(state)
            action_pad = self.pad_action(action)

            if is_offline:
                teacher_q1, teacher_q2 = teacher.critic(state, action)
            else:
                next_state_pad = self.pad_state(next_state)
                next_action = self.actor_target(next_state_pad)

                noise = (
                        torch.rand_like(next_action) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

                next_action = next_action[:, :action_dim_org]
                next_action_pad = self.pad_action(next_action)

                target_q1, target_q2 = self.critic_target(next_state_pad, next_action_pad)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + not_done * self.discount * target_q

        current_q1, current_q2 = self.critic(state_pad, action_pad)
        if is_offline:
            critic_loss = F.mse_loss(current_q1, teacher_q1) + F.mse_loss(current_q2, teacher_q2)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        else:
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            self.critic_optimizer_online.zero_grad()
            critic_loss.backward()
            self.critic_optimizer_online.step()

        loss = [None, critic_loss.cpu().data.numpy()]

        if is_offline or self.total_it[idx] % self.policy_freq == 0:

            current_action = self.actor(state_pad)[:, :action_dim_org]
            current_action_pad = self.pad_action(current_action)

            actor_loss_t = -teacher.critic.Q1(state, current_action)

            if is_offline:
                actor_loss = actor_loss_t.mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
            else:
                actor_loss = -self.critic.Q1(state_pad, current_action_pad)
                actor_loss = 1.0 * actor_loss + 1.0 * actor_loss_t
                actor_loss = actor_loss.mean()
                self.actor_optimizer_online.zero_grad()
                actor_loss.backward()
                self.actor_optimizer_online.step()

            self.update_target_network()

            loss[0] = actor_loss.cpu().data.numpy()

        return loss
