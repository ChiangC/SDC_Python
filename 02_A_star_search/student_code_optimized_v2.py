import math

#f = g(path length) + h(estimated distance)

#计算两点之间的距离(path cost)
def dist_between(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def heuristic_cost_estimate(point, pGoal):
    return dist_between(point, pGoal)

def reconstruct_path(cameFrom, current):
    total_path = [current]
    while current in cameFrom.keys():
        current = cameFrom[current]
        total_path.append(current)
    total_path.reverse()    
    return total_path
        
def shortest_path(M,start,goal):
    print("shortest path called")
    
    if(start != goal):

        # The set of nodes already evaluated
        closedSet = set()

        # The set of currently discovered nodes that are not evaluated yet.
        # Initially, only the start node is known.
        openSet = {start}
        

        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many nodes, cameFrom will eventually contain the
        # most efficient previous step.
        cameFrom = {}

        # For each node, the cost of getting from the start node to that node.
        gScore = {}

        # The cost of going from start to start is zero.
        gScore[start] = 0

        # For each node, the total cost of getting from the start node to the goal
        # by passing by that node. That value is partly known, partly heuristic.
        fScore = {}

        # For the first node, that value is completely heuristic.
        fScore[start] = heuristic_cost_estimate(M.intersections[start], M.intersections[goal])
       
        while openSet:
            current = min(fScore.keys(), key=(lambda k: fScore[k]))  
            print("current:", current)
            print("openSet:", openSet)
            print("fScore:", fScore)
            
            print("========")
            if current == goal:
                return reconstruct_path(cameFrom, current)

            openSet.remove(current)
            closedSet.add(current)
            fScore.pop(current)
            
            for neighbor in M.roads[current]:
                if neighbor in closedSet:
                    continue          # Ignore the neighbor which is already evaluated.

                if neighbor not in openSet:  # Discover a new node
                    openSet.add(neighbor)

                # The distance from start to a neighbor
                #the "dist_between" function may vary as per the solution requirements.
                tentative_gScore = gScore[current] + dist_between(M.intersections[current], M.intersections[neighbor])
                if (neighbor in gScore) and tentative_gScore >= gScore[neighbor]:
                    continue          # This is not a better path.

                # This path is the best until now. Record it!
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = gScore[neighbor] + heuristic_cost_estimate(M.intersections[start], M.intersections[goal])

        return [start, goal]
    
    else:
        return [goal]

    
#Jiangchun
#chiangchuna@gmail.com
#June 8, 2019  12:50

