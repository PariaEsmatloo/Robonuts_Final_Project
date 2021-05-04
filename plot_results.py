import matplotlib.pyplot as plt
import csv



num_episodes =200


mm_REINFORCE = []
ss_REINFORCE =[]

with open('REINFORCE.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(row)
        mm_REINFORCE.append(float(row[1]))
        ss_REINFORCE.append(float(row[2]))
        
        
mm_AC = []
ss_AC =[]

with open('ActorCritic.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(row)
        mm_AC.append(float(row[1]))
        ss_AC.append(float(row[2]))
        
        
mm_rand = []
ss_rand =[]

with open('RandomAgent.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(row)
        mm_rand.append(float(row[1]))
        ss_rand.append(float(row[2]))
        
        
                        
m_plus_std_REINFORCE = []
m_minus_std_REINFORCE =[]

for (mm , ss) in zip ( mm_REINFORCE, ss_REINFORCE):
    m_plus_std_REINFORCE.append(mm+ss)
    m_minus_std_REINFORCE.append(mm-ss)

m_plus_std_AC = []
m_minus_std_AC =[]

for (mm , ss) in zip ( mm_AC, ss_AC):
    m_plus_std_AC.append(mm+ss)
    m_minus_std_AC.append(mm-ss)

m_plus_std_rand = []
m_minus_std_rand =[]

for (mm , ss) in zip ( mm_rand, ss_rand):
    m_plus_std_rand.append(mm+ss)
    m_minus_std_rand.append(mm-ss)
        
        
        
        
fig, ax = plt.subplots() 
# plt.figure(figsize = [30,24], dpi=100)  
     
R_m =ax.plot(range(num_episodes), mm_REINFORCE)
R_s = ax.fill_between(range(num_episodes),m_minus_std_REINFORCE , m_plus_std_REINFORCE , alpha=0.2)

AC_m =ax.plot(range(num_episodes), mm_AC)
AC_s = ax.fill_between(range(num_episodes),m_minus_std_AC , m_plus_std_AC , alpha=0.2)

rand_m =ax.plot(range(num_episodes), mm_rand)
rand_s =ax.fill_between(range(num_episodes),m_minus_std_rand , m_plus_std_rand , alpha=0.2)


ax.set(xlabel='Episode', ylabel='Total Score',
       title='Reacher-v2 Performance Using 3 Agents' )

# ax.legend([(R_m,R_s)],["REINFORCE"],loc=2)
ax.legend(['REINFORCE', 'Actor-Critic', 'Random'])
ax.grid()


fig.savefig('Compare_3_agents.png',dpi=300)
plt.show()    





# for (mm , ss) in zip ( m_score, std_score):
#     m_plus_std.append(mm+ss)
#     m_minus_std.append(mm-ss)

# ax.fill_between(range(num_episodes),m_minus_std , m_plus_std , alpha=0.2)

# ax.set(xlabel='Episode', ylabel='Total Score',
#        title='Averaged Random Agent Results')
# ax.grid()


# fig.savefig(fig_filename_average)
# plt.show()    
