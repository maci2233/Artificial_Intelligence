import pygame
import pygame.freetype
from pynput.keyboard import Controller
import random
import numpy as np
import matplotlib.pyplot as plt
import os

class Note:
    def __init__(self, bar=None, type=None):
        if not bar: #THESE ARE THE NOTES THAT FALL
            self.type = random.randint(0, 3)
            self.y = 100
            self.color = colors[self.type]
        else: #THESE ARE THE CIRCLES INSIDE THE BAR
            self.type = type
            self.y = bar.y + bar.height//2
            self.color = bar_colors[self.type]
        self.size = 25 #circle radius
        self.x = notes_pos_x[self.type]

    def move(self):
        self.y += 1

    def draw(self):
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.size)

class Bar:
    def __init__(self, width, height, y):
        self.x = 0
        self.y = y
        self.width = width
        self.height = height
        self.color = (200, 200, 200)
        #THE BAR WILL CONTAIN 4 NOTES INSIDE JUST FOR VISUALIZATION
        self.notes = [Note(self, 0), Note(self, 1), Note(self, 2), Note(self, 3)]

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        for i, note in enumerate(self.notes):
            note.draw()
            self.notes[i].color = bar_colors[i]

def delete_notes(): #DELETE NOTES THAT PASSED THE WINDOW LIMIT IN Y
    for i, notes_t in enumerate(notes): #iterate for all lists of note types
        if len(notes_t) > 0:
            note = notes_t[0]
            if note.y > height:
                notes_t.pop(0)


def check_key(key): #key is the number that was pressed (1, 2, 3, 4)
    #we check if there is a note in the bar that corresponds to the pressed key
    k = key-1
    try:
        note = notes[k][0]
        bar_2 = bar.height//2
        if note.y >= bar.y + bar_2 - note_press_pixel_error and note.y <= bar.y + bar_2 + note_press_pixel_error:
            notes[k].pop(0)
            return True
        else:
            return False
    except IndexError:
        return False

    #bar.notes[k].color = bar_color_pressed

def get_observation():
    obs_res = [0, 0, 0, 0]
    for i, notes_t in enumerate(notes): #iterate for all lists of note types
        if len(notes_t) > 0:
            note = notes_t[0]
            #if note.y >= bar.y and note.y <= bar.y + bar.height:
            bar_2 = bar.height//2
            if note.y >= bar.y + bar_2 - note_press_pixel_error and note.y <= bar.y + bar_2 + note_press_pixel_error:
                obs_res[i] = 1
    return obs_res

def action(ind, obs): #IND IS THE INDEX HIGHEST Q-VALUE CORRESPONDING TO THE CURRENT OBSERVATION OF THE ENVIRONMENT
    reward = 0
    if ind == 0: #PRESS NOTHING
        sum_obs = sum(obs)
        if sum_obs == 0:
            return KEY_NOT_PRESSED_CORRECT_REWARD
        else:
            return sum_obs * KEY_NOT_PRESSED_WRONG_REWARD
    else: #PRESS THE KEY THAT CORRESPONDS TO THE ACTION
        #print('action: ', action_list[ind])
        for ch in action_list[ind]:
            i = int(ch) - 1 #i will be used as the index for the obs values
            keyboard.press(ch)
            keyboard.release(ch)
            #IF THE KEY THAT WAS PRESSED IS NOT INSIDE THE BAR THERE MUST BE A NEGATIVE REWARD
            #ELSE WE RETURN A POSITIVE REWARD
            if obs[i] == 0:
                reward += KEY_PRESSED_WRONG_REWARD
            else:
                reward += KEY_PRESSED_CORRECT_REWARD
        for i, note in enumerate(obs):
            if note == 1 and str(i+1) not in action_list[ind]:
                reward += KEY_NOT_PRESSED_WRONG_REWARD
        #print(f'reward: {reward}')
        return reward


pygame.init()
GAME_FONT = pygame.freetype.SysFont('Consolas', 15)
colors = [(255, 20, 0),(255, 255, 30), (20, 255, 0), (20, 0, 255)] #NOTE COLORS THAT WILL FALL
notes_pos_x = [100, 200, 300, 400] #THE POSITION IN X OF THE NOTES THAT WILL FALL
bar_colors = [(255, 80, 80),(240, 250, 90), (60, 230, 90), (60, 110, 220)] #NOTE COLORS INSIDE THE BAR
bar_color_pressed = (255, 80, 200) #BAR NOTE COLOR CHANGES WHEN PRESSED (FUNCTIONALITY NOT ADDED ATM)
bg_color = (0, 0, 0)
width = 500
height = 800
NEW_NOTE_EVERY = 100 #EVERY X iterations WE ADD A NEW NOTE, A LOWER VALUE MEANS NOTES APPEAR FASTER
NOTE_TYPES = 4 #THERE ARE 4 TYPE OF NOTES
EPISODES = 12 #ITERATIONS TO TRAIN THE AGENT
KEY_PRESSED_WRONG_REWARD = -10 #REWARD FOR HITTING A KEY WHEN IT WAS NOT NEEDED
KEY_PRESSED_CORRECT_REWARD = 5 #REWARD FOR HITTING A KEY WHEN IT WAS NEEDED
KEY_NOT_PRESSED_CORRECT_REWARD = 0 #REWARD FOR HITTING NOTHING WHEN IT WAS NOT NEEDED
KEY_NOT_PRESSED_WRONG_REWARD = -10 #REWARD FOR HITTING NOTHING WHEN IT WAS NEEDED
note_press_pixel_error = 1 #We use this constanst to define a +- limit in the pixel error to have a correct note pressed
SHOW_EVERY = 1 #EVERY X ITERATIONS A VIDEO OF THE AGENT PLAYING WILL APPEAR
LEARNING_RATE = 0.05
DISCOUNT = 0.9

action_list = ['', '1', '2', '3', '4', '12', '13', '14', '23', '24', '34']
notes = [[] for _ in range(NOTE_TYPES)] #LIST OF NOTES, EVERY LIST INSIDE WILL CONTAIN ONLY ONE TYPE OF NOTES
bar = Bar(width, 100, 650)
#bar = Bar(width, 60)
keyboard = Controller()

load_q_table = False #If we already trained an agent we can set this to True and load the q_table
q_table_file = 'guitar_hero_table.pickle'

if not load_q_table:
    q_table = {}
    #WE HAVE A FOR LOOP FOR EACH NOTE TYPE, BASICALLY WE ARE ADDING ALL THE POSSIBLE
    #COMBINATIONS OF THE GAME STATUS TO THE Q TABLE, FOR EXAMPLE THE KEY (0, 1, 0, 0)
    #IN THE Q TABLE MEANS THAT THERE IS A NOTE TYPE=2 THAT IS INSIDE THE BAR SO THEREFORE
    #THE KEY NUMBER 2 SHOULD BE PRESSED BY THE AGENT
    for note_1_in_bar in range(2):
        for note_2_in_bar in range(2):
            for note_3_in_bar in range(2):
                for note_4_in_bar in range(2):
                    q_table[(note_1_in_bar, note_2_in_bar, note_3_in_bar, note_4_in_bar)] = [np.random.uniform(-5, 0) for _ in range(11)]
else:               #THERE ARE 5 POSSIBLE ACTIONS THAT START WITH RANDOM Q-VALUES: press-1, press-2, press-3, press-4, press nothing
    with open(q_table_file, "rb") as f:
        q_table = pickle.load(f)
#print(q_table)
episode_rewards = []
episode_mistakes = []
for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        show = True
    else:
        show = False
    episode_reward = 0
    episode_mistake = 0
    #i = 0
    run = True
    if show:
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Agent Hero")
        pygame.display.flip()
        clock = pygame.time.Clock()
    #while run:
    #This i represents how many iterations the EPISODE will last
    #Increasing this value means the agent will have more time to
    #learn during each EPISODE
    for i in range(6000):
        delete_notes() #here we delete the notes that have passed the window limit
        obs = tuple(get_observation())
        next_action = np.argmax(q_table[obs])
        reward = action(next_action, obs)
        episode_reward += reward
        if reward < 0:
            episode_mistake += 1
        curr_q_val = q_table[obs][next_action]
        new_obs = tuple(get_observation())
        max_future_q = np.max(q_table[new_obs])
        #new_q = reward
        new_q  = (1 - LEARNING_RATE) * curr_q_val + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][next_action] = new_q

        clock.tick(2000)
        for event in pygame.event.get(): #All the events that are happening (list)
            if event.type == pygame.QUIT:
                run = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    check_key(1)
                if event.key == pygame.K_2:
                    check_key(2)
                if event.key == pygame.K_3:
                    check_key(3)
                if event.key == pygame.K_4:
                    check_key(4)
        if i % NEW_NOTE_EVERY == 0:
            notes_created = []
            for _ in range(np.random.randint(1, 3)):
                while True:
                    note = Note()
                    if note.type in notes_created:
                        continue
                    notes[note.type].append(note)
                    break
        screen.fill(bg_color)
        bar.draw()
        for notes_col in notes:
            for note in notes_col:
                note.move()
                note.draw()
        #if i == 4000:
            #run = False
        if show:
            GAME_FONT.render_to(screen, (100, 22), f"Mistakes: {episode_mistake}        Reward: {episode_reward}", (255, 255, 255))
            pygame.display.flip()
        if not run:
            break
    if show:
        pygame.quit()
    print(f"episode  {episode+1}: reward -> {episode_reward}\tmistakes -> {episode_mistake}")
    episode_rewards.append(episode_reward)
    episode_mistakes.append(episode_mistake)


x_axis = [i+1 for i in range(len(episode_rewards))]
plt.plot(x_axis, episode_rewards, label='Score')
plt.plot(x_axis, episode_mistakes, label='Mistakes')
plt.title("Improvement over time")
#plt.ylabel(f"episode info")
plt.xlabel("episode #")
plt.legend()
plt.show()
