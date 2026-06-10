# Local Spatial Hierarchy - Second Stage of HES-COMA Framework

This section presents how the second stage of the HES-COMA framework uses strategic position movements derived from the global spatial hierarchy to learn tactical actions within the local spatial hierarchy. This hierarchy utilizes the results learned from the global spatial hierarchy. Figure 4 illustrates how the local spatial hierarchy learns tactical actions. Unlike the first stage which focuses only on movement, the second stage employs an environment in which tactical actions can be executed to facilitate the learning of tactical behaviors. The state data include not only positional information but also additional details specific to tactical action execution.

> **Figure 4.** Structure of the local spatial hierarchy.

---

## Action Space Architecture

The framework operates with two distinct action spaces that enable hierarchical decision-making:

### Global Action Space

Consists of **nine discrete actions**:
- **Eight directional movement actions:** North, Northeast, East, Southeast, South, Southwest, West, Northwest
- **One stationary action:** "stop"

These actions are designed to handle **macro-level strategic positioning**, where agents learn to navigate to optimal spatial locations based on energy field guidance. Global actions operate at a higher temporal resolution, making decisions at every game step to ensure continuous spatial optimization.

### Local Action Space

Encompasses **five tactical actions** that are executed only when the global hierarchy determines optimal positioning:
- **Shooting** — attempting to score
- **Stealing** — intercepting opponent ball possession
- **Rebounding** — retrieving missed shots
- **Blocking** — defending against opponent shots
- **Passing** — transferring ball to teammates

These actions represent **micro-level tactical decisions** that require precise timing and situational awareness, as they involve direct interaction with game objects and opponents.

---

## Hierarchical Control Mechanism

Figure 5 illustrates the hierarchical control mechanism of HES-COMA, showing the sequential decision-making process and the interaction between global and local action spaces.

> **Figure 5.** HES-COMA hierarchical control mechanism.

The control mechanism operates through a **sequential decision-making process** where global actions take precedence over local actions:

1. At each time step, the **global agent** first evaluates the current position using energy field information and determines whether to move to a better position or remain stationary.

2. If any **directional movement** is selected, the corresponding movement is executed immediately, and local actions are suppressed for that time step.

3. **Local actions are triggered exclusively** when the global agent selects the "stop" action, indicating that the agent has reached a strategically advantageous position.

This hierarchical control flow ensures that tactical actions are only attempted from strategically sound positions, thereby improving action effectiveness and success rates.

---

## Integration with Global Hierarchy

Using an already trained model, the global spatial hierarchy determines the strategic position movement (Global Action), effectively reducing the agent's state-action space and simplifying the learning process. As the energy field is critical for global-level learning, it is used exclusively within the global spatial hierarchy.

When the global spatial hierarchy selects a **"stop" action** at a specific point, the local spatial hierarchy decides on a tactical action based on the current game situation. Otherwise, the movement action selected by the global hierarchy proceeds to the final action.

This separation of information between hierarchies:
- Minimizes unnecessary observation data
- Reduces computational load at the local level
- Helps agents learn tactical actions efficiently from advantageous positions

---

## Reward Structure

Given the nature of sports games, each action's success or failure is tied to scoring or conceding points, which then translate into rewards. The local spatial hierarchy:

- **Gains a reward** for scoring
- **Receives a penalty** for conceding
- **Receives an additional reward** if a tactical action succeeds appropriately
- **Incurs a penalty** if the action is inappropriate or fails

With this reward structure, the second stage of learning involves:
- Global spatial hierarchy assigning strategic positions
- Local spatial hierarchy focusing on tactical actions at those positions

This approach dramatically narrows the state-action space. Consequently, by having the global spatial hierarchy direct spatial movement and the local spatial hierarchy learn tactical actions, the learning complexity is reduced, and the overall gameplay performance is enhanced.

---

## Training Process - Second Stage

Figure 6 shows the pseudocode for the second-stage training of the HES-COMA. In the second stage, the environment $\varepsilon_l$ is designed such that **LAgent** performs local actions only when **GAgent** chooses the "stop" action, thereby allowing GAgent and LAgent to divide their roles and optimize cooperative behavior.

The training process proceeds over $N$ episodes as before:

1. In each episode, starting from the initialized environment $\varepsilon_l$, **LAgent** utilizes partial observation $o^l$ to select actions $a^l$ following its policy $\pi^l_{\theta^l}$.

2. Meanwhile, **GAgent** executes a global action $a^g$ based on its already trained policy $\pi^g_{\theta^g}$.

3. If GAgent performs a **movement action**, LAgent's action is ignored.

4. Only when GAgent chooses **"stop"** is LAgent's local action actually applied.

5. In each step, the state, observation, action, reward, next state, and termination signal are recorded in a **local-level buffer**.

6. After an episode ends, the policy parameters of the LAgent are updated based on the **COMA algorithm**.

The learning rates for LAgent's critics and actors were set to:
- **Local spatial Critic Learning Rate**
- **Local spatial Actor Learning Rate**

The overall training method followed that of the first stage, with the only difference being the content stored in the buffer.