import math



#f = (path length) + (estimated distance)

#计算两点之间的距离(path cost)
def path_length_between_2_points(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

#所有的点到目标点的预测距离(estimated distance to goal)
def estimated_distances_2_goal(intersections, goal):
    estimated_distances_dict = {}
    for key in intersections:
        estimated_distances_dict[key] = path_length_between_2_points(intersections[key], goal)
    return estimated_distances_dict

def estimated_distance_2_goal(point, pGoal):
    return path_length_between_2_points(point, pGoal)


#===================================================================
# Node Start
#===================================================================
#路径节点类
class Node(object):
    
    def __init__(self):
        #路径末端状态:1 表示已经到达终点
        self.state = 0
        
        #当前路径的长度
        self.currentPathLen = 0.0 
        
        #总代价
        self.cost = 0.0
        
        #指向其他节点
        self.parent = None
     
        #节点下标
        self.pIndex = -1
        
    def __str__(self):
        path_points = []
        pNode = self.parent
        while pNode != None:
            path_points.append(pNode.pIndex)
            pNode = pNode.parent
        path_points.reverse()  
        return "PathPoints: %s, Cost： %f" %(str(path_points), self.cost)
    
#===================================================================
# Node End
#===================================================================

#===================================================================
# Frontier Start
#===================================================================
#边缘类
class Frontier(object):
    
    def __init__(self):
        #前缘节点
        self.frontierPoints = {}
        #前缘节点中cost最小的节点
        self.minCostFPoint = None
    
    #添加新的前缘节点
    def addPoint(self,fNode):
        if fNode == None:
            return
        #当新的前缘节点没有拓展到终点，且前缘节点中cost最小的点已经拓展到终点，且新的前缘节点cost要比minCostFPoint的cost大，很显然没有必要添加到frontier
        if (self.minCostFPoint != None) and (self.minCostFPoint.state) == 1 and (self.minCostFPoint.cost < fNode.cost):
            return
        
        if fNode.pIndex in self.frontierPoints:
            oldNode = self.frontierPoints.get(fNode.pIndex)
            if fNode.cost < oldNode.cost:
                self.frontierPoints[oldNode.pIndex] = fNode
        else:
            self.frontierPoints[fNode.pIndex] = fNode
            self.__sortDicByCost()

            

    #弹出最小cost值的前缘节点
    def popMinCostPoint(self):
        minCostP = self.minCostFPoint;
        if self.minCostFPoint != None:
            self.frontierPoints.pop(self.minCostFPoint.pIndex)
            self.__sortDicByCost()
        return minCostP
    
    #以cost，从小到大进行排序
    def __sortDicByCost(self):
        sortedDic = sorted(self.frontierPoints.items(),key=lambda dict_original:dict_original[1].cost,reverse=False);
#         print("Sorted Dic:",sortedDic)
        if len(sortedDic) > 0:
            self.minCostFPoint = sortedDic[0][1]
        else:
            self.minCostFPoint = None
    
#===================================================================
# Frontier End
#===================================================================    
def getPathByNode(node):
    path_points = []
    if node != None:
        pNode = node
        while pNode != None:
            path_points.append(pNode.pIndex)
            pNode = pNode.parent
        path_points.reverse()
        
    return path_points
        
def shortest_path(M,start,goal):
    print("shortest path called")
    
    if(start != goal):
        #计算所有点到目标点的预测距离
        #estimated_distances_dict = estimated_distances_2_goal(M.intersections, M.intersections[goal])

        #前缘
        frontier = Frontier()
        
        #已探索列表
        exploredSet = set()
        
        #初始化起点
        startNode = Node()
        startNode.pIndex = start
        
        frontier.addPoint(startNode) #把起点添加到frontier

        possibleMinCostPoint = None
        
        while len(frontier.frontierPoints) > 0:
            
            #获取cost最小的前缘节点
            minCostFPoint = frontier.popMinCostPoint()
            
            if (minCostFPoint != None and minCostFPoint.state == 1) and (possibleMinCostPoint != None and minCostFPoint.cost < possibleMinCostPoint.cost):
                possibleMinCostPoint = minCostFPoint
                
            #添加到已探索列表
            if minCostFPoint != None:
                exploredSet.add(minCostFPoint.pIndex)
                
            #cost值最小的前缘节点可拓展的交叉点    
            roads = M.roads[minCostFPoint.pIndex] 

            for i in range(len(roads)):
                #1.不能拓展已经搜索过的交叉点;
                if not(roads[i] in exploredSet):
                    #新的前缘节点
                    newFNode = Node()
                    newFNode.state = 1 if roads[i] == goal else 0
                    newFNode.pIndex = roads[i]
                    newFNode.parent = minCostFPoint
                    
                    #起点开始，到该节点的路径总长度
                    newFNode.currentPathLen = minCostFPoint.currentPathLen + path_length_between_2_points(M.intersections[roads[i]], M.intersections[minCostFPoint.pIndex])
                    
                    #cost = pathLen + estimatedDistance2Goal
                    newFNode.cost = newFNode.currentPathLen + estimated_distance_2_goal(M.intersections[roads[i]], M.intersections[goal])
                    
                    #添加前缘节点到fontier 
                    frontier.addPoint(newFNode) 
                    
                    print(str(newFNode))  
                    #第一次最先拓展到终点的节点
                    if newFNode.state == 1 and possibleMinCostPoint == None:   
                        possibleMinCostPoint = newFNode  
                        
        #END while===========================================================================
        
        return getPathByNode(possibleMinCostPoint)
    
    else:
        return [goal]

#Jiangchun
#chiangchuna@gmail.com
#June 8, 2019  12:50

