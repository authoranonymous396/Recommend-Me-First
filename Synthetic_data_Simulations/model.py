import numpy as np
import itertools
import random 
import copy

debug = False


class User:
    def __init__(self, config, id):
        self.config = config
        self.id = id

        # the best CC followed so far
        self.best_followed_CC = None

    def decide_follow(self, c):
        '''Evaluates whether the user wants to follow CC c.

        input: c - a content creator
        ------
        output: bool - decision if it follows c'''

        # it follows c iff they are better (closer to top) then the best followed so far
        if (self.best_followed_CC is None) or (self.best_followed_CC.id > c.id):
            self.best_followed_CC = c
            return True

        return False


class CC:
    def __init__(self, config, id):
        self.config = config
        self.id = id


class Network:
    '''Class capturing a follower network between from users to items.
    In this version of the code we assumme that each item is a content creator/channel.
    '''

    def __init__(self, config, G=None, groups = None, favorite=None):
        self.config = config

        num_users = config['num_users']
        num_CCs = config['num_CCs']
        num_groups = config['num_groups']

        self.G = G
        self.groups = groups 

        if self.G is None:
            self.G = np.zeros((num_users, num_CCs), dtype=bool)
        if self.groups is None: 
            # create the group of users randomly 
            users = np.arange(num_users)
            np.random.shuffle(users)
            group_size = num_users//num_groups
            group_left = num_users%num_groups

            self.groups = np.zeros((num_users, num_groups), dtype=int)
            counter = 0
            for i in range(num_groups):
                g_size = group_size + (1 if i <group_left else 0)
                group_users = users[counter:counter+g_size]
                self.groups[group_users, i] = 1
                counter += g_size

            
    

        self.num_followers = np.count_nonzero(self.G, axis=0)
        self.num_followees = np.count_nonzero(self.G, axis=1)

        self.num_followers_per_group = [] 

        for p in range(self.config['num_groups']):
            users_in_group = np.where(self.groups[:, p] == 1)[0]
            num_followers_in_p = self.G[users_in_group, :]

            self.num_followers_per_group.append( np.count_nonzero(num_followers_in_p, axis = 0))

    def follow(self, u, c, num_timestep, when_users_found_best):
        '''User u follows content creator c; and updates the Network

        input: u - user
               c - CC
               num_timestep - the iteration number of the platform (int)
               when_users_found_best - a list of length the number of users who keeps the timesteps when each of the user found their best CC (or -1 if they didn't yet)
        '''

        if not self.G[u.id][c.id]:
            if u.decide_follow(c):
                self.G[u.id][c.id] = True
                self.num_followers[c.id] += 1
                self.num_followees[u.id] += 1
                group = np.argmax(self.groups[u.id, :])
                self.num_followers_per_group[group][c.id] += 1 
                

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
    
    def aggregate_groups(self):
        # pairwise aggregation of groups 
        old_group = self.groups
        old_num_groups = self.config['num_groups']
        new_groups = np.zeros((self.config['num_users'], old_num_groups//2), dtype=int)
        new_followers_per_group = []
        count = 0
        for i in range( old_num_groups//2):
            new_groups[:,i] = self.groups[:, count]+ self.groups[:, count+1]
            new_followers_per_group.append(self.num_followers_per_group[count]+self.num_followers_per_group[count+1])
            count+=2
        self.config['num_groups'] = old_num_groups//2
        self.groups = new_groups
        self.num_followers_per_group = new_followers_per_group
        if debug:
            print("There were", old_num_groups," now ", self.config['num_groups'])

    def aggregate_all_groups(self):
        #aggregate all groups into one
        old_group = self.groups
        old_num_groups = self.config['num_groups']
        new_groups = np.zeros((self.config['num_users'], 1), dtype=int)

        followers_per_group = np.zeros(self.config['num_CCs'])
        for i in range(old_num_groups):
            new_groups[:, 0] += self.groups[:, i]
            followers_per_group += self.num_followers_per_group[i]
    

        self.config['num_groups'] = 1
        self.groups = new_groups
        self.num_followers_per_group = [followers_per_group]


class RS:
    '''Class for the Recommender System (i.e., descoverability  procedure).
    '''

    def __init__(self, config, content_creators):
        self.config = config


    def recommend_general(self, content_creators, num_followers_per_group, groups):
            ''' input: content_creators - a list of content creators
                    num_followers - a numpyarray with the probability of choosing each CC
            -----
            output: a CC chosen based on PA'''

        
            num_users = self.config['num_users']
            num_CCs = self.config['num_CCs']
            alpha = self.config['alpha']
  
            prob_choice_per_group = (num_followers_per_group + np.ones(num_CCs))**alpha 
            prob_choice_per_group = [group/np.sum(group) for group in prob_choice_per_group]

            recommendations = np.zeros(num_users, dtype= object)
            for g in range(groups.shape[1]):
                num_users_g = np.count_nonzero(groups[:,g])
                recommendations_g = self.config['random_generator'].choice(content_creators, num_users_g, p=prob_choice_per_group[g])
                recommendations[groups[:,g]!= 0] = recommendations_g
            
            return recommendations

    def recommend_random(self, content_creators,num_followers_per_group, groups):
        ''' input: content_creators - a list of content creators
        -----
        output: a list of recommendations of CC chosen uniformly at ranodm'''

        num_users = self.config['num_users']
        return self.config['random_generator'].choice(content_creators, num_users)

    

    def recommend_Comparaison(self, content_creators, num_followers_per_group, groups):
        num_users = self.config['num_users']
        num_CCs = self.config['num_CCs']
        alpha = self.config['alpha']
        
        recommendations_time_1 = np.zeros(num_users, dtype= object)
        recommendations_time_2 = np.zeros(num_users, dtype= object)
        
        first= 1 
        second = 2
        for g in range(int ( groups.shape[1]/2 ) ):
            # Suppose we have m groups where m = (n 2), and the permutations are ordered (1, 2)... (1, m), (2, 3)... (2, m), ..., (m-1, m)
            pair_CCs = np.random.permutation([first, second]) #randomly permute the pair of CCs
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

    def recommend(self, content_creators, num_followers_per_group, groups):
        '''A rapper that choses the appropriate RS.

        input: content_creators - a list of content creators
               num_followers - a numpyarray with the probability of choosing each CC
        -----
        output: a list of reccommendations (one per user)'''

        if self.config['rs_model'] == 'UR':
            return self.recommend_random(content_creators, num_followers_per_group, groups)
        elif self.config['rs_model'] == 'general':
            return self.recommend_general(content_creators, num_followers_per_group, groups)
    


class Platform:
    def __init__(self, config):
        self.config = config

        # the platform keeps track of the number of timesteps it has been iterated
        self.timestep = 0

        self.network = Network(config)
        self.users = [User(config, i)
                      for i in range(config['num_users'])]
        self.CCs = [CC(config, i)
                    for i in range(config['num_CCs'])]
        self.RS = RS(config, self.CCs)
        
        self.next_RS = None 
        # keep track of the timesteps when users found their best CC
        self.users_found_best = [-1 for u in self.users]
        # keep track of the position of the recommended CC in the ranking of the user
        # self.users_rec_pos = []
        # keep track of the average quality experienced by users
        self.average_pos_best_CC = []

        # the users who did not converged yet
        self.id_searching_users = list(range(self.config['num_users']))
        self.current_shows = None 

        if debug:
            print('Generated users and CCs.')
      

    def iterate(self):
        '''Makes one iteration of the platform.
        Used only to update the state of the platform'''

        # 0) the platform starts the next iteration
        num_users = self.config['num_users']
        self.timestep += 1

        if self.timestep in self.config['time_agg']:
            self.network.aggregate_groups()
        # 1) each user gets a recommendation
        if self.RS.config['rs_model']== 'Comparison_random':
            if self.timestep == 1:
                recs1, recs2 = self.RS.recommend_Comparaison(self.CCs, self.network.num_followers_per_group, self.network.groups)
                recs = recs1
                self.next_RS = recs2
            elif self.timestep ==2:
                recs = self.next_RS
                self.network.aggregate_all_groups()
            else:
                recs = self.RS.recommend_random(self.CCs, self.network.num_followers_per_group, self.network.groups)
        elif self.RS.config['rs_model']=='Comparison_popularity':
            if self.timestep == 1:
                recs1, recs2 = self.RS.recommend_Comparaison(self.CCs, self.network.num_followers_per_group, self.network.groups)
                recs = recs1
                self.next_RS = recs2
            elif self.timestep ==2:
                recs = self.next_RS
                self.network.aggregate_all_groups()
            else:
                recs = self.RS.recommend_general(self.CCs, self.network.num_followers_per_group, self.network.groups)
        else:
            recs = self.RS.recommend(self.CCs, self.network.num_followers_per_group, self.network.groups)
        # record the position of the recommended CC
        # self.users_rec_pos.append([c.id for i, c in enumerate(recs)])

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
            average_pos += u.best_followed_CC.id / num_users
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
            filter(lambda i: self.users[i].best_followed_CC.id != 0, self.id_searching_users))
    def check_convergence(self):
        # the platform converged if there are no more searching users (users who can find better CCs)
        return len(self.id_searching_users) == 0
