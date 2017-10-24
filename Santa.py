


# Read Data:

import numpy as np
import random,math
import pandas as pd
import dill

# read data
datadir = "D:\Google Drive\Code\R\Code\Santa\Data\\"
Data = pd.read_csv(datadir + 'gifts.csv')

# Set Data parameters
locations, weights  = (Data.values)[:,1:3] , (Data.values)[:,3]
max_weight = 990
north_pole = np.array([90,0])
weight_limit = 990
sleigh_weight = np.array([10]).reshape(1,1)
NrTrips = math.ceil(np.sum(weights)/990)
NrGifts = len(weights)


# Function for Initial Solution:



# Get total weight of the Trip
def total_weight(solution):
    return np.sum(weights[solution])


# Calculate the distance for string
def distance(bitstring):
    # Convert to numpy
    Sol = np.array(list(bitstring),dtype=int).reshape(NrGifts,NrTrips)
    # Check if each gift is assigned once and only once
    if not all(np.sum(Sol,axis=1) == 1):
        return 1e100
    # Check if max weight condition has exceeded for any Trip
    elif any(np.sum(Sol * weights[:,np.newaxis],axis=0) > max_weight):
        return 1e100
    # find locations of each Trip and their area using shoelace formula after sorting
    else:
        Area = 0
        for ind in range(0,NrTrips):
            points = locations[list(np.where(Sol[:,ind,None] == 1)[0])]
            pts = points - np.mean(points,0)
            ord = np.argsort(list(map(math.atan2,pts[:,0],pts[:,1])))
            x, y = points[ord,0], points[ord,1]
            Area += 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        return Area



# Starting solution based only on capacity
def initial_sol():
    Trips = []
    gifts_pos = np.array([i for i in range(NrGifts)])
    while len(gifts_pos) > 0:
        select_pos = random.choice(gifts_pos)
        for Trip in Trips:
            if total_weight(Trip+ [select_pos]) <= max_weight:
                Trip.append(select_pos)
                gifts_pos = np.delete(gifts_pos,np.argwhere(gifts_pos==select_pos))
                break
        else:
            Trips.append([])
    return Trips,gifts_pos




# Run multiple times and get best initial solution
bestSol = 1e500
Niter = 1

print('Finding initial Solution...')
for _ in range(0,Niter):
    temp,t = initial_sol()
    # Convert solution to string format
    NrTrips = len(temp)
    Trip = np.zeros(shape = (NrGifts,NrTrips),dtype = int)
    for j in range(0,len(temp)):
        for i in range(0,len(temp[j])):
            Trip[temp[j][i],j] = 1
    STemp = "".join([str(i) for i in Trip.flatten()])
    # Print the solution distance and and update best solution if its better
    print('iteration ' + str(_) + ":" + str(distance(STemp)))
    if distance(STemp) < bestSol:
        STrip = STemp
        bestSol = distance(STemp)
        NrTrips = len(temp)

# Save session
dill.dump_session(datadir + 'SantaSol.pkl')
print('Found initial Solution.')




# find neighbour solutions for Annealing
def newSol(bitstring,prob):
    # Randomly choose a gift and assignment it to another trip if it meets the probability condition
    if random.random() < prob :
        i = random.choice(range(0,NrGifts))
        gift = np.array(list(bitstring[i*NrTrips:(i+1)*NrTrips]),dtype=int)
        index = np.where(gift == 1)[0][0]
        nindex = random.choice(list(np.where(gift == 0)[0]))
        gift[[index,nindex]] ^= 1
        return bitstring[:i*NrTrips] + ''.join([str(i) for i in gift]) + bitstring[(i+1)*NrTrips:]
    # Randomly choose two trips and swap gifts among them
    else:
        gift = np.array(list(bitstring),dtype=int).reshape(NrGifts,NrTrips)
        i,j = random.sample(range(0,NrTrips),2)
        trip1, trip2 = gift[:, i, None],gift[:, j, None]
        if any(trip1-trip2 == 1) and any(trip1-trip2 == -1):
            ind = [random.choice(list(np.where(trip1-trip2 == 1)[0])),random.choice(list(np.where(trip1-trip2 == -1)[0]))]
            trip1[ind] ^= 1
            trip2[ind] ^= 1
            gift[:, i, None],gift[:, j, None] = trip1, trip2
        return ''.join([str(i) for i in gift.flatten()])

# Annealing for assigning gifts to Trips without considering the order of the gifts
def assignTrip(bitstring,prob,temperature_begin= 1.0e+200, temperature_end=.1, cooling_factor=.9999, nb_iterations =2):
    trip_best = bitstring
    distance_best = distance(trip_best)
    distances_current = []
    distances_best = []
    ids_iteration = {}
    step = 0
    # Random start multiple times
    for iteration in range(nb_iterations):
        # the search is restarted at every iteration from the best known solution
        temperature = temperature_begin
        trip_current = trip_best[:]
        distance_current = distance_best
        distance_new = distance_best
        trip_new = trip_best[:]
        # Start the loop using temperature
        while temperature >= temperature_end:
            n = 0
            while n <= nb_iterations * 2:
                trip_new = newSol(trip_current,prob)
                # use probability to decide whether update required
                distance_current = distance(trip_current)
                distance_new = distance(trip_new)
                diff = distance_new - distance_current
                # check whether there is an improvement
                if diff < 0 or  math.exp( -diff / temperature ) > random.random():
                    trip_current = trip_new[:]
                    distance_current = distance_new
                else:
                    # reset trip and distance
                    distance_new = distance_current
                    trip_new = trip_current[:]
                # update the best if current solution is better  not part of the annealing itself, just used for the restart
                if distance_current < distance_best:
                    trip_best = trip_current[:]
                    distance_best = distance_current
                # update iterators
                step = step + 1
                n = n + 1
            temperature = temperature * cooling_factor
            ids_iteration[float(distance_best)] = trip_best
    return trip_best,distance_best,ids_iteration




# Calling the annealing method
print('Started Assigning Gifts to Trips...')
assign_Sol,assign_Dist,assign_Dict = assignTrip(STrip,0.2,temperature_begin= 1.0e+200,cooling_factor=.9999, nb_iterations =2)

# Convert solution to do anneal for order
Sol = np.array(list(assign_Sol),dtype = int).reshape(NrGifts,NrTrips)

# Save session
dill.dump_session(datadir + 'SantaSol.pkl')
print('Assigned Gifts to Trips.')




# vectorize and the distance covered by each trip

def haversine_np_array(loc1,loc2):
    lat1, lon1, lat2, lon2 = loc1[:,0], loc1[:,1], loc2[:,0], loc2[:,1]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km.reshape(len(loc1),1)


def weighted_trip_length_np(location,weights):
    # add north pole to the last and second array to create pairs
    loc = np.append(north_pole.reshape(1,2),location,axis=0)
    loc_nxt = np.roll(loc,-1,axis=0)

    # add sleigh weight and subtract the delivered gifts
    weight = np.append(weights.reshape(len(weights),1),sleigh_weight,axis=0)
    weight = np.append([np.sum(weight)],np.sum(weight) - np.cumsum(weight),axis=0)[:-1]


    # total distance
    return np.dot(weight,haversine_np_array(loc,loc_nxt))


# better Order
def assignOrder(gifts,temperature_begin= 1.0e+200, temperature_end=.1, cooling_factor=.9999, nb_iterations =2):

    trip_best = list(gifts)
    distance_best = weighted_trip_length_np(locations[trip_best],weights[trip_best])

    distances_current = []
    distances_best = []
    ids_iteration = {}

    step = 0
    # Random start multiple times
    for iteration in range(nb_iterations):
        # the search is restarted at every iteration from the best know solution
        temperature = temperature_begin
        trip_current = trip_best[:]
        distance_current = distance_best
        distance_new = distance_best
        trip_new = trip_best[:]

        # Start the loop using temperature
        while temperature >= temperature_end:
            n = 0
            while n <= nb_iterations * 2:
                [i,j] = sorted(random.sample(range(len(trip_current)),2))
                trip_new = trip_current.copy()
                trip_new[i], trip_new[j] = trip_new[j], trip_new[i]

                # use probability to decide whether update required
                distance_current = weighted_trip_length_np(locations[trip_current],weights[trip_current])
                distance_new = weighted_trip_length_np(locations[trip_new],weights[trip_new])
                diff = distance_new - distance_current

                # check whether there is an improvement
                if diff < 0 or  math.exp( -diff / temperature ) > random.random():
                    trip_current = trip_new[:]
                    distance_current = distance_new
                else:
                    # reset trip and distance
                    distance_new = distance_current
                    trip_new = trip_current[:]

                # update the best if current solution is better  not part of the annealing itself, just used for the restart
                if distance_current < distance_best:
                    trip_best = trip_current[:]
                    distance_best = distance_current

                # update iterators
                step = step + 1
                n = n + 1

            temperature = temperature * cooling_factor
            ids_iteration[float(distance_best)] = list(trip_best)
    return list(trip_best),distance_best





# Run SA for every trip and append results to master data
print('Started Ordering Gifts in the Trips...')
df = pd.DataFrame()
for OTrip in range(Sol.shape[1]):
    print("Annealing Started for trip " +str(OTrip))
    gifts = np.where(Sol[:,OTrip,None] == 1)[0]
    Ord, Dist = assignOrder(gifts)
    df = df.append(pd.DataFrame([[str(Ord),np.float64(Dist),OTrip]],columns = ['Gifts','Distance','TripId']))
    print("Annealing Ended for trip " +str(OTrip))

# Save session
dill.dump_session(datadir + 'SantaSol.pkl')
print('Solution Found.')




# File formatting and write to csv
Sub = pd.concat([pd.Series(row['TripId'], row['Gifts'].replace("[","").replace("]","").split(',')) for _, row in df.iterrows()]).reset_index()
Sub.columns = ["GiftId","TripId"]
Sub.GiftId = [int(i) + 1 for i in Sub.GiftId]
Sub.to_csv(datadir  + "\\Santa_Solution.csv",index = False)




# Compute actual Score
north_pole = (90,0)
weight_limit = 1000
sleigh_weight = 10



from haversine import haversine

def weighted_trip_length(stops, weights):
    tuples = [tuple(x) for x in stops.values]
    # adding the last trip back to north pole, with just the sleigh weight
    tuples.append(north_pole)
    weights.append(sleigh_weight)

    dist = 0.0
    prev_stop = north_pole
    prev_weight = sum(weights)
    for location, weight in zip(tuples, weights):
        dist = dist + haversine(location, prev_stop) * prev_weight
        prev_stop = location
        prev_weight = prev_weight - weight
    return dist

def weighted_reindeer_weariness(all_trips):
    uniq_trips = all_trips.TripId.unique()

    if any(all_trips.groupby('TripId').Weight.sum() > weight_limit):
        raise Exception("One of the sleighs over weight limit!")

    dist = 0.0
    for t in uniq_trips:
        this_trip = all_trips[all_trips.TripId==t]
        dist = dist + weighted_trip_length(this_trip[['Latitude','Longitude']], this_trip.Weight.tolist())

    return dist



gifts = pd.read_csv(datadir  + '\\gifts.csv')
sample_sub = Sub

all_trips = sample_sub.merge(gifts, on='GiftId')

FinalScore = weighted_reindeer_weariness(all_trips)
print(FinalScore)

# Save session
dill.dump_session(datadir + 'SantaSol.pkl')

