import numpy as np
import itertools
import random 
import copy
import networkx as nx
import community as community_louvain
debug = False


class User:
    def __init__(self, config, id):
        self.config = config
        self.id = id
        
        # the best CC followed so far
        self.best_followed_CC = None
        self.alpha = config['alphas'][id]
        self.quality_user = config['quality_users'][id]
        self.range = config['num_CCs']
        if config['uniform']: 
            self.prob_less = config['probability'] # probability to follow the recommendation 
        else : 
            self.prob_less = config['probability'] 
            self.range = config['range']
        #self.num_interest = random.choices(range(1, config['num_genres']+1),weights = [0.8, 0.2], k=1)[0]
        #self.num_interest = random.randint(1, config['num_genres'])
        self.num_interest = 1
        self.interests = random.sample(range(1, config['num_genres']+1), self.num_interest)

    def decide_follow(self, c):
        #compute prob to follow the cc
        if c.genre not in self.interests:
            return False
        if self.best_followed_CC is None:
            prob = 1
        else:
            prob = 0
            if (self.best_followed_CC.id > c.id):
                prob += self.alpha* self.prob_less #maximizer 
            if ((self.best_followed_CC.id > self.quality_user) and (self.best_followed_CC.id> c.id)): 
                prob += (1-self.alpha)*self.prob_less #satisfier
        
        if np.random.rand() <= prob:
            self.best_followed_CC = c
            return True 
        else: 
            if np.random.rand() <=prob and ((c.id - self.best_followed_CC.id) < self.range):
                return True
            return False 
class CC:
    def __init__(self, config, id):
        self.config = config
        self.id = id

        sum = 0
        for i in range(len(config['proportion_CCs'])):
            sum += config['proportion_CCs'][i]
            if self.id < sum * config['num_CCs']:
                self.genre = i+1 
                break 


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
            # create the groups randomly 
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
                # if c is the top CC for u and u is a satisfier 
                if c.id < u.quality_user: 
                    when_users_found_best[u.id] = num_timestep  
                    # input()
                return 1
            return -1 
        else : 
            return None 

    def is_following(self, u, c):
        return self.G[u.id][c.id]
    
    def aggregate_groups(self):
        #pairwise 
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
    
    def recommend_perfect_fairness(self, content_creators, num_followers_per_group, groups):

        num_users = self.config['num_users']
        num_CCs = self.config['num_CCs']
        k = int (num_users / self.config['num_groups'])

        followed = np.array([k * (2 * num_CCs - i - 1) for i in range(1, num_CCs+1)])
        want = np.array ([k/i * ((num_CCs-i) * (num_CCs - i - 1)) for i in range(1, num_CCs+1)])

        recs = np.zeros(num_users, dtype= object)
        for u in self.users:
            CC = u.best_followed_CC 
            need = num_CCs-3
            while recs[u.id] == 0:
                if (need < CC.id ) and (want[need]>0): 
                    recs[u.id] = content_creators[need]
                    want [need] -= 1
                need -= 1
        return recs
    def recommend_SAR(self, content_creator, signal_matrix, signal_matrix_time,similarity_metric = "Jaccard"):
        num_users = self.config['num_users']
        num_CCs = self.config['num_CCs']
        recs = np.zeros(num_users, dtype= object)
        # Compute the item-item similarity matrix
        only_positive = signal_matrix.copy()
        #print(only_positive[0])
        only_positive[only_positive == -1] = 0
        cooccurrence_matrix = np.dot(only_positive.T, only_positive)
        S = np.zeros((num_CCs, num_CCs))
        if similarity_metric == "Jaccard":
            for i in range(num_CCs):
                for j in range(num_CCs):
                        if cooccurrence_matrix[i, j] != 0:
                            S[i, j] = cooccurrence_matrix[i, j] / (cooccurrence_matrix[i, i] + cooccurrence_matrix[j, j] - cooccurrence_matrix[i, j])
        if similarity_metric == "lift":
            for i in range(num_CCs):
                for j in range(num_CCs):
                        if cooccurrence_matrix[i, j] != 0:
                            S[i, j] = cooccurrence_matrix[i, j] / (cooccurrence_matrix[i, i] * cooccurrence_matrix[j, j])
        if similarity_metric == "counts":
            S = cooccurrence_matrix
        # Compute user affinity scores 
        user_affinity_scores = signal_matrix.copy()
        timestep = len(signal_matrix_time)
        for user in range(num_users):
            for CC in range(num_CCs):
                decay_factors = np.array([(1/2)**(timestep-t) for t in range(len(signal_matrix_time))])
                signal_values = np.array([signal_matrix_time[t][user, CC] for t in range(len(signal_matrix_time))])
                signal_values [signal_values == -1] = 0
                user_affinity_scores[user, CC] = np.sum(signal_values * decay_factors)

        # Compute the recommendation scores
        recommendation_scores = np.dot(user_affinity_scores, S)
        max_values = np.max(recommendation_scores, axis=1, keepdims=True)
        print(max_values[:20])
        max_mask = recommendation_scores == max_values
        max_mask = max_mask.astype(float)
        #random_values = np.random.random(recommendation_scores.shape) * max_mask
        random_values =self.config['random_generator'].random(recommendation_scores.shape) * max_mask

        recs_index = np.argmax(random_values, axis=1)
        recs = np.array(content_creator)[recs_index]
        print(recs_index[:20])
        return recs
        

    def recommend_collaborative_filtering (self, content_creators, signal_matrix): 
        num_users = self.config['num_users']
        num_CCs = self.config['num_CCs']
        recs = np.zeros(num_users, dtype= object)

        scores_matrix  = np.dot(signal_matrix, signal_matrix.T)
        np.fill_diagonal(scores_matrix, -float('inf'))
      
        for user in range(num_users):
            scores = scores_matrix[user]
       
            max_score = np.max(scores)
            max_users = np.where(scores == max_score)[0]

            unseen_by_user  = np.where(signal_matrix[user] == 0)[0]
            if len(unseen_by_user) == 0:
                #user has already been recommanded everything
                recs[user] = self.config['random_generator'].choice(content_creators)
            else :  
                mat = signal_matrix[np.ix_(max_users, unseen_by_user)]
                numerator = np.sum(mat == 1, axis=0)
                denominator = np.sum(mat == 1, axis=0) + np.sum(mat == -1, axis=0)
                CC_scores = np.zeros_like(denominator, dtype=float) 

                mask = denominator > 0
                CC_scores[mask] = numerator[mask] / denominator[mask]

                max_CC_score = np.max(CC_scores)
                CC_indices = np.where(CC_scores == max_CC_score)[0]
                recommandation = np.random.choice(CC_indices)
                recommendation_indice = unseen_by_user[recommandation]
                recs[user] = content_creators[recommendation_indice]
        return recs
    def hybrid_collaborative_popularity(self, content_creators, signal_matrix, num_followers_per_group, groups, weight = 1/2):
        num_users = self.config['num_users']
        num_CCs = self.config['num_CCs']
        recs = np.zeros(num_users, dtype= object)

        scores_matrix  = np.dot(signal_matrix, signal_matrix.T)
        np.fill_diagonal(scores_matrix, -float('inf'))

        prob_choice_per_group = (num_followers_per_group + np.ones(num_CCs))
        scores_popularity = [group/np.sum(group) for group in prob_choice_per_group][0]
        for user in range(num_users):
            scores_cf = np.zeros(num_CCs) 
            scores = scores_matrix[user]
       
            max_score = np.max(scores)
            max_users = np.where(scores == max_score)[0]

            unseen_by_user  = np.where(signal_matrix[user] == 0)[0]
            if len(unseen_by_user) == 0:
                #user has already been recommanded everything
                scores_cf = (1/num_CCs) * np.ones(num_CCs) 
            else :  
                mat = signal_matrix[np.ix_(max_users, unseen_by_user)]
                numerator = np.sum(mat == 1, axis=0)
                denominator = np.sum(mat == 1, axis=0) + np.sum(mat == -1, axis=0)
                CC_scores = np.zeros_like(denominator, dtype=float) 

                mask = denominator > 0
                CC_scores[mask] = numerator[mask] / denominator[mask]
                #CC_scores  =  np.sum(mat == 1, axis=0)/ (np.sum(mat == 1, axis=0)+ np.sum(mat == -1, axis=0))
                min_score = np.min(CC_scores)
                max_score = np.max(CC_scores)
            
                if max_score > min_score:
                    normalized_scores = (CC_scores - min_score) / (max_score - min_score)
                else:
                    normalized_scores = np.ones_like(CC_scores) / len(CC_scores)
                for i, idx in enumerate(unseen_by_user):
                    scores_cf[idx] = normalized_scores[i]
                
            combined_scores = weight * scores_cf + (1 - weight) * scores_popularity
            max_score = np.max(combined_scores)
            best_indices = np.where(combined_scores == max_score)[0]
            if len(best_indices) > 1:
                best_indices = self.config['random_generator'].choice(best_indices)
            else : 
                best_indices = best_indices[0]
            recs[user] = content_creators[best_indices]
        return recs
                   
    def hybrid_collaborative_ranking(self, content_creators, signal_matrix, ranking):
        num_users = self.config['num_users']
        num_CCs = self.config['num_CCs']
        recs = np.zeros(num_users, dtype= object)

        scores_matrix  = np.dot(signal_matrix, signal_matrix.T)

        np.fill_diagonal(scores_matrix, -num_users*num_CCs)

        for user in range(num_users):
            scores = scores_matrix[user]
       
            max_score = np.max(scores)
            max_users = np.where(scores == max_score)[0]

            unseen_by_user  = np.where(signal_matrix[user] == 0)[0]
            if len(unseen_by_user) == 0:
                #user has already been recommanded everything
                recs[user] = self.config['random_generator'].choice(content_creators)
            else :  
                mat = signal_matrix[np.ix_(max_users, unseen_by_user)]
                numerator = np.sum(mat == 1, axis=0)
                denominator = np.sum(mat == 1, axis=0) + np.sum(mat == -1, axis=0)
                CC_scores = np.zeros_like(denominator, dtype=float) 

                mask = denominator > 0
                CC_scores[mask] = numerator[mask] / denominator[mask]

                max_CC_score = np.max(CC_scores)
                CC_indices = np.where(CC_scores == max_CC_score)[0]
                recommendation_indices = unseen_by_user[CC_indices]
                #Take the CC with the best position in ranking 
                ranking_indices = ranking[recommendation_indices]
                best_ranking_index = np.argmin(ranking_indices)
                recommendation_indice = recommendation_indices[best_ranking_index]
                recs[user] = content_creators[recommendation_indice]
        return recs


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
        self.ranking = np.zeros(config['num_CCs']) # 1 community 
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
   
        self.signal_matrix = np.zeros((config['num_users'], config['num_CCs']))
        self.signal_matrix_time = []
    def iterate(self):
        '''Makes one iteration of the platform.
        Used only to update the state of the platform'''
        # 0) the platform starts the next iteration
        num_users = self.config['num_users']
        self.timestep += 1
        if self.timestep in self.config['time_agg']:
            self.network.aggregate_groups()
        # 1) each user gets a recommendation
        elif self.RS.config['rs_model']== 'Comparaison':
            if self.timestep == 1:
                recs1, recs2 = self.RS.recommend_Comparaison(self.CCs, self.network.num_followers_per_group, self.network.groups)
                recs = recs1
                self.next_RS = recs2
            elif self.timestep ==2:
                recs = self.next_RS
                self.network.aggregate_all_groups()
            else:
                recs = self.RS.recommend_random(self.CCs, self.network.num_followers_per_group, self.network.groups)
                #recs = self.RS.recommend_collaborative_filtering(self.CCs, self.signal_matrix)
                #recs = self.RS.recommend_general(self.CCs, self.network.num_followers_per_group, self.network.groups)
                #recs = self.RS.hybrid_collaborative_popularity(self.CCs, self.signal_matrix, self.network.num_followers_per_group, self.network.groups)
        elif self.RS.config['rs_model']== 'perfect_fairness':
            if self.timestep == 1:
                recs1, recs2 = self.RS.recommend_Comparaison(self.CCs, self.network.num_followers_per_group, self.network.groups)
                recs = recs1
                self.next_RS = recs2
            elif self.timestep ==2:
                recs = self.next_RS
                self.network.aggregate_all_groups()
            else:
                recs = self.RS.recommend_perfect_fairness(self.CCs, self.network.num_followers_per_group, self.network.groups)
            
        elif self.RS.config['rs_model']== 'collaborative_filtering':
            recs = self.RS.recommend_collaborative_filtering(self.CCs, self.signal_matrix)
        elif self.RS.config['rs_model']== 'collaborative_filtering_SAR':
            recs = self.RS.recommend_SAR(self.CCs, self.signal_matrix, self.signal_matrix_time)
        elif self.RS.config['rs_model']== 'collaborative_filtering_SAR_comp':
            if self.timestep == 1:
                recs1, recs2 = self.RS.recommend_Comparaison(self.CCs, self.network.num_followers_per_group, self.network.groups)
                recs = recs1
                self.next_RS = recs2
            elif self.timestep ==2:
                recs = self.next_RS
                self.network.aggregate_all_groups()
            else:
                recs = self.RS.recommend_SAR(self.CCs, self.signal_matrix, self.signal_matrix_time)
     
        elif self.RS.config['rs_model']== 'hybrid_collaborative_popularity':
            recs = self.RS.hybrid_collaborative_popularity(self.CCs, self.signal_matrix, self.network.num_followers_per_group, self.network.groups)
        elif self.RS.config['rs_model']== 'hybrid_collaborative_ranking':
            if self.timestep == 1:
                recs1, recs2 = self.RS.recommend_Comparaison(self.CCs, self.network.num_followers_per_group, self.network.groups)
                recs = recs1
                self.next_RS = recs2
            elif self.timestep ==2:
                recs = self.next_RS
                self.network.aggregate_all_groups()
            elif self.timestep == 3:
                #find communities of CCs 
                com_idx = np.zeros((self.config['num_CCs'], self.config['num_CCs']))
                for i in range(self.config['num_CCs']):
                    for j in range(self.config['num_CCs']):
                        if i != j:        
                            sub_matrix = self.signal_matrix[:, [i, j]]
                        
                            followed_both = sub_matrix[np.all(sub_matrix == [1,1], axis=1)].shape[0]
                            follow_only_i = sub_matrix[np.all(sub_matrix == [1,-1], axis=1)].shape[0]
                            follow_only_j = sub_matrix[np.all(sub_matrix == [-1,1], axis=1)].shape[0]
        
                            if followed_both /(follow_only_i + follow_only_j+followed_both) >0.1:
                                com_idx[i, j] = 1
                                com_idx[j, i] = 1        


                G = nx.from_numpy_array(com_idx)     
                partition = community_louvain.best_partition(G)
                print(partition)
                follow_counts = np.sum(self.signal_matrix == 1, axis=0)
                sorted_indices = np.argsort(-follow_counts)
                self.ranking = sorted_indices
                recs = self.RS.hybrid_collaborative_ranking(self.CCs, self.signal_matrix, self.ranking)
            else:
                #recs = self.RS.recommend_collaborative_filtering(self.CCs, self.signal_matrix, weight = 1/3)
                recs = self.RS.hybrid_collaborative_ranking(self.CCs, self.signal_matrix, self.ranking)
        else:
            recs = self.RS.recommend(self.CCs, self.network.num_followers_per_group, self.network.groups)
        # record the position of the recommended CC
        # self.users_rec_pos.append([c.id for i, c in enumerate(recs)])

        # 2) each user decides whether or not to follow the recommended CC
        for u in self.users:
            decision = self.network.follow(
                u, recs[u.id], self.timestep, self.users_found_best)
            if decision !=  None : 
                self.signal_matrix[u.id, recs[u.id].id] = decision 
            if self.config['rs_model'] == 'SAR':
                if decision !=  None : 
                    self.signal_matrix[u.id, recs[u.id].id] = decision 
                else:
                    self.signal_matrix[u.id, recs[u.id].id] = -1
                self.signal_matrix_time.append(self.signal_matrix.copy())
        # 3) if we run until convergence, update the searching users
        if self.config['num_steps'] == 0:
            self.update_searching_users()

        # record the average CC position experienced by CCs
        average_pos = np.zeros(self.config['num_genres'])
        num_users = self.config['num_users']
        num_users_per_comm = np.zeros(self.config['num_genres'])
        for u in self.users: 
            for i in u.interests:
                num_users_per_comm[i-1] += 1
        for u in self.users:
            if u.num_interest == 1: 
                if u.best_followed_CC != None: # bc of communities 
                    average_pos[u.interests[0]-1] += u.best_followed_CC.id / num_users_per_comm[u.interests[0]-1]
                else: 
                    average_pos[u.interests[0]-1] += self.config['num_CCs']  / num_users_per_comm[u.interests[0]-1]
            else: 
                if u.best_followed_CC != None :
                    i = u.best_followed_CC.genre 
                    average_pos[i-1] += u.best_followed_CC.id  / num_users_per_comm[i-1]
                else:
                    for i in u.interests:
                        average_pos[i-1] += self.config['num_CCs'] / num_users_per_comm[i-1]
        self.average_pos_best_CC = np.concatenate((self.average_pos_best_CC, average_pos)).tolist()
    

    def update_searching_users(self):
        '''Updates the list of users who are still searching for the best CC.
        i.e. those who did not find the best CC out of the ones that could be recommended
        '''
        self.id_searching_users = list(
            filter(lambda i: self.users[i].best_followed_CC.id != 0, self.id_searching_users))
    def check_convergence(self):
        # the platform converged if there are no more searching users (users who can find better CCs)
        return len(self.id_searching_users) == 0
