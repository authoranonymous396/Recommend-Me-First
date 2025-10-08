import numpy as np



def get_if_IF(popularity):
    num_CCs = len(popularity)

    # get the inexes of CCs by the number of followers
    popularity = np.array(popularity)
    ordered_by_popualrity = np.flip(popularity.argsort())
    id_to_position_popualrity = {i:pos for pos, i in enumerate(ordered_by_popualrity)}
    # print(ordered_by_popualrity)

    # find if fair (CC_i is in terms of popularity on the position it deserves or more twoards the top)
    def is_fair(i):
        return int(id_to_position_popualrity[i] <= i)
    
    return [is_fair(i) for i in range(num_CCs)]

def get_if_IF_multiple_communities (popularity,community = 1, range_prop = [1]):
    '''Given a list of follower counts for content creators in the order of their
    quality finds if each is treated fairly. 
    
    Input
    ------
    popularity: list 
        List of nonegative integers corresponding to follower counts for each creator
    ------

    Ouput 
    ------
    is_fair: list
        A list of 0/1 of same length as the input. 
    '''

    num_CCs = len(popularity)
    current_cc = 0
    genres = []
    for n_com in range(community):
        # Calculate how many CCs should be in this community
        cc_count = int(range_prop[n_com] * num_CCs)
        # Assign this genre to the appropriate number of CCs
        for _ in range(cc_count):
            if current_cc < num_CCs:
                genres.append(n_com + 1)
                current_cc += 1
   #last ommunity
    popularity = np.array(popularity)
    genres = np.array(genres)
    fairness = []
    for n_com in range(community):  
        ids = np.where(np.array(genres) == n_com+1)[0]
        popularity_genres= popularity[ids]
        ordered_by_popularity = np.flip(popularity_genres.argsort())
        #sorted_indices = np.flip(np.argsort(popularity_genre))
        id_to_position_popularity = {i:pos for pos, i in enumerate(ordered_by_popularity)}
        fairness_com  = [int (id_to_position_popularity[i]<=i) for i in range(len(id_to_position_popularity))]
        fairness.append(fairness_com)
    return fairness


def get_epsilon_IF (popularity, epsilon = 1):
    '''Given a list of follower counts for content creators in the order of their 
    quality finds if each is treated "fairly
    
    Input
    ------
    popularity: list
        List of nonnegative integers corresponding to follower counts for each creators
    epsilon: integer
        Integer corresponding to how many places we accept to be "wrongly" fair
    ------
    
    Output 
    ------
    is fair: list 
        A list of 0/1of same length as input 
    '''
    
    num_CCs = len(popularity)

    popularity = np.array(popularity)
    ordered_by_popularity = np.flip(popularity.argsort())
    id_to_position_popularity = {i: pos for pos, i in enumerate(ordered_by_popularity)}

    def is_fair(i):
        # Check if the content creator's position deviates by at most epsilon from their fairness position
        return int(id_to_position_popularity[i]<= i + epsilon)
    
    return [is_fair(i) for i in range(num_CCs)]


def get_merit_IF(popularity, num_comparaison = 5):
  '''Given a list of follower counts for content creators in the order of their 
    quality finds if each is treated "fairly
    
    Input
    ------
    popularity: list
        List of nonnegative integers corresponding to follower counts for each creators
    num_comparaison: integer
        Integer corresponding to number of content creators to compare with 
    ------
    
    Output 
    ------
    is fair: list 
        A list of 0/1of same length as input 
    '''
  num_CCs = len(popularity)

  popularity = np.array(popularity)

  def compare_with_worse_ccs(i):
    #Find how many content creators have more followers than i but have lower quality 
    num_worse_ccs = 0
    for j in range(i+1, num_CCs):
      if popularity[j] > popularity[i]:
        num_worse_ccs += 1
    if num_worse_ccs >= num_comparaison:
      return 0
    return 1

  return [compare_with_worse_ccs(i) for i in range(num_CCs)]

def agg_fairness_IF(popularity):
   if_ccs =  get_if_IF (popularity)
   return np.mean(if_ccs)



