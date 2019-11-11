import gym
import numpy as np
import matplotlib.pyplot as plt

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


env = gym.make("MountainCar-v0")
#env.reset()


learning_rate = 0.1
discount = 0.95 #que tan importante son nuestras futuras acciones.
epochs = 25000
show_every = 500

#discrete size sera de largo 2 porque el environment contiene la posicion y la velocidad, osea 2 features
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) #generalmente estos son random o eso parece
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

epsilon = 0.5 #Mientras mas grande hay mas probabilidad de que se haga una accion aleatoria para explorar
start_epsilon_decaying = 1
end_epsilon_decaying = epochs // 2
epsilon_decay_value = epsilon/(end_epsilon_decaying - start_epsilon_decaying)


#En este caso el q_table sera de 20x20 para tener todas las combinaciones de posicion y velocidad
#env.action_space.n es para que al tener las 20x20 combinaciones tengamos un q-value para cada accion posible del agente
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
epoch_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

for epoch in range(epochs):
    epoch_reward = 0
    if epoch % show_every == 0:
        print(epoch)
        render = True
    else:
        render = False
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        epoch_reward += reward
        epoch_rewards.append(reward)
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state]) #Obtenemos el q value mas alto
            current_q = q_table[discrete_state + (action, )]  #agregarle el (action, ) hace que en lugar de obtener los 3 q values solo obtengamos el del action que se hizo
            new_q = (1-learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position: #new_state[0] al parecer es la posicion del carro entonces si es mayor a la posicion de la meta le pones un reward de 0
            q_table[discrete_state + (action, )] = 0
        discrete_state = new_discrete_state
    if end_epsilon_decaying >= epoch >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value
    epoch_rewards.append(epoch_reward)
    if not epoch % show_every:
        avg_reward = sum(epoch_rewards[-show_every:])/len(epoch_rewards[-show_every:])
        aggr_ep_rewards['ep'].append(epoch)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(epoch_rewards[-show_every:]))
        aggr_ep_rewards['max'].append(max(epoch_rewards[-show_every:]))
env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()
