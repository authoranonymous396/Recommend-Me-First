import numpy as np
import copy
import ast 
debug = False    
class User:
    def __init__(self, engagement, id):
        self.id = id
        # the best CC followed so far
        self.best_followed_CC = None 
        self.best_followed_CC_id = engagement

    def decide_follow(self, c):
        if (self.best_followed_CC_id is None)or (self.best_followed_CC_id > c.id) :
            self.best_followed_CC = c
            self.best_followed_CC_id = c.id
            return True
        return False    

class CC:
    def __init__(self, id):
        self.id = id

class Network:
    '''Class capturing a follower network between from users to items.
    In this version of the code we assumme that each item is a content creator/channel.
    '''

    def __init__(self, config, G, groups = None, favorite=None):
        self.config = config

        self.num_users = config['num_users']
        self.num_CCs = config['num_CCs']
        self.G = G # network 
        self.groups=None # random groups of users
        self.num_groups = config['num_groups']
        self.config = config 

        if self.G is None: 
            self.G = np.zeros((self.num_users, self.num_CCs), dtype=bool)
        if self.config['rs_model'] == 'Comparison':
            if self.groups is None: # randomly assig users to one group
                users = np.arange(self.num_users)
                np.random.shuffle(users)
                group_size = self.num_users//self.num_groups
                group_left = self.num_users%self.num_groups
                self.groups = np.zeros((self.num_users, self.num_groups), dtype=int)
                counter = 0
                for i in range(self.num_groups):
                    g_size = group_size + (1 if i <group_left else 0)
                    group_users = users[counter:counter+g_size]
                    self.groups[group_users, i] = 1
                    counter += g_size
        self.num_followers = np.count_nonzero(self.G, axis=0)
        self.num_followees = np.count_nonzero(self.G, axis=1)   
        if debug: 
            print(self.num_followers)
    def follow(self, u, c, num_timestep, when_users_found_best):
        '''User u follows content creator c; and updates the Network

        input: u - user
               c - CC
               num_timestep - the iteration number of the platform (int)
               when_users_found_best - a list of length the number of users who keeps the timesteps when each of the user found their best CC (or -1 if they didn't yet)
        '''
        if not self.G[u.id, c.id]:
            if u.decide_follow(c):
                self.G[u.id][c.id] = True
                self.num_followers[c.id] += 1
                self.num_followees[u.id] += 1

                # if c is the top CC, then u found their best CC this round
                if c.id == 0:
                    when_users_found_best[u.id] = num_timestep
                if debug:
                    print("       ", num_timestep, ": " ,u.id, " folllows ", c.id,
                            ", when_users_found_best becomes ", when_users_found_best,
                            ", and num_followers is ", self.num_followers)
                    # input()
    def is_following(self, u, c):
        return self.G[u.id][c.id]
    def aggregate_all_groups(self):
        #aggregate all groups into one
        old_group = self.groups
        old_num_groups = self.num_groups
        new_groups = np.zeros((self.config['num_users'], 1), dtype=int)

        for i in range(old_num_groups):
            new_groups[:, 0] += self.groups[:, i]

        self.num_groups = 1
        self.groups = new_groups
    
class RS: 
    '''Class for the Recommender System (i.e., descoverability procedure).
    '''
    def __init__(self, config, content_creators): 
        self.config = config
        self.content_creators = content_creators
        self.num_CCs = len(content_creators)
    
    def recommend_general(self, content_creators, num_followers):
        ''' input: content_creators - a list of content creators
                num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a CC chosen based on PA'''

    
        num_users = self.config['num_users']
        num_CCs = self.config['num_CCs']
        alpha = self.config['alpha']

        prob_choice = (num_followers + np.ones(num_CCs))**alpha
        prob_choice /= sum(prob_choice)
        if debug:
            print('Prob choice RS:', prob_choice)

        
        return self.config['random_generator'].choice(content_creators, num_users, p=prob_choice) 
    def recommend_random(self, content_creators):
        ''' input: content_creators - a list of content creators
                num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a CC chosen based on PA'''

       
        num_users = self.config['num_users']
        return self.config['random_generator'].choice(content_creators, num_users)
    def recommend_Comparaison(self, content_creators, groups):
        ''' input: content_creators - a list of content creators
                num_followers - a numpyarray with the probability of choosing each CC
        '''
        num_users = self.config['num_users']
        num_CCs = self.config['num_CCs']
        alpha = self.config['alpha']
        
        recommendations_time_1= np.zeros(num_users, dtype=object)
        recommendations_time_2 = np.zeros(num_users, dtype=object)

        first = 1
        second = 2
        for g in range(int(groups.shape[1]/2 )):
            pair_CCs = np.random.permutation([first, second]) # randomly permute the pair of CCs
            #pair_CCs = [first, second] #keep the order of the pair of CCs 
            #pair_CCs = [second, first]  #worst order of the pair of CCs 
            r1 = pair_CCs[0]
            r2 = pair_CCs[1]

            recommendations_time_1[groups[:,g]!= 0] = content_creators[r1-1]
            recommendations_time_2[groups[:,g]!= 0] = content_creators[r2-1]


            recommendations_time_1[groups[:,int(g+ (num_CCs*(num_CCs-1))/2) ]!= 0] = content_creators[r2-1]
            recommendations_time_2[groups[:,int(g+ (num_CCs*(num_CCs-1))/2) ]!= 0] = content_creators[r1-1]

            second += 1
            if (second > num_CCs):
                first += 1
                second = first+1
        return recommendations_time_1, recommendations_time_2
    def recommend(self, content_creators, num_followers):
        '''A rapper that choses the appropriate RS.

        input: content_creators - a list of content creators
               num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a list of reccommendations (one per user)'''

        if self.config['rs_model'] == 'UR':
            return self.recommend_random(content_creators)
        elif self.config['rs_model'] == 'general':
            return self.recommend_general(content_creators,num_followers)
class Platform: 
    def __init__(self, config):
        self.config = config
        self.timestep = 0
        engagement = ast.literal_eval(config['engagement'].strip('"'))
        G = np.where(engagement is None, None, engagement)
        engagement = [idx[0] if idx.size > 0 else None
                for idx in (np.where(np.array(row) == 1)[0] for row in engagement)]
        if debug: 
            print(G [0])
            #print("Engagement matrix  is: \n", G)
            print("Engagement sum per movie: \n", G.sum(axis = 0))
            print("Engagement matrix shape  is: \n", G.shape)


        # Initialize users and content creators
        self.network = Network(config, G= G)
        self.users = [User(engagement[i], i) for i in range(config['num_users'])]
        self.CCs = [CC(i) for i in range(config['num_CCs'])]
        self.RS = RS(config, self.CCs)
        # delete the engagement bc it takes too much memory 
        del engagement
        del config['engagement']  # Remove from config too

        # for comparaison, keep next recommendations 
        self.next_RS = None 
        #keeè track of the timesteps when users found their best CC and other stats 
        self.users_found_best = [-1 for u in self.users]
        self.average_pos_best_CC = []
        self.id_searching_users = list(range(self.config['num_users']))
        self.current_shows = None 
    
    def iterate(self):
        '''Makes one iteration of the platform. 
        Used only to update the state of the platform'''
        # 0) the platform starts the next iteration
        num_users = self.config['num_users']
        self.timestep += 1
        
        # 1) each user gets a recommendation 
        if self.RS.config['rs_model'] == 'Comparison':
            if self.timestep == 1:
                recs1, recs2 = self.RS.recommend_Comparaison(self.CCs, self.network.groups)
                recs = recs1
                self.next_RS = recs2
            elif self.timestep == 2:
                recs = self.next_RS
                self.network.aggregate_all_groups() # aggregate all groups into one
            else:
                recs = self.RS.recommend_general(self.CCs, self.network.num_followers)
                
        else: 
            recs = self.RS.recommend(self.CCs, self.network.num_followers)
        # 2) each user decides whether or not to follow the recommended CC
        for u in self.users:
            self.network.follow(
                u, recs[u.id], self.timestep, self.users_found_best)

        # 3) if we run until convergence, update the searching users
        if self.config['num_steps'] == 0:
            self.update_searching_users()

        # record the average CC position experienced by CCs
        average_pos = 0
        num_users = self.config['num_users']
        for u in self.users:
            average_pos += u.best_followed_CC_id / num_users
        self.average_pos_best_CC.append(average_pos)

        if debug:
            print('Recommendations: ', [r.id for r in recs])
            print('New network:', self.network.G)
            print('Number of followers:', self.network.num_followers)
            print('Number of followees:', self.network.num_followees)

    def update_searching_users(self):
        '''Updates the list of users who are still searching for the best CC.
        i.e. those who did not find the best CC out of the ones that could be recommended
        '''
        self.id_searching_users = list(
            filter(lambda i: self.users[i].best_followed_CC_id != 0, self.id_searching_users))
    def check_convergence(self):
        # the platform converged if there are no more searching users (users who can find better CCs)
        return len(self.id_searching_users) == 0

    