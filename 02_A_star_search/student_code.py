import math
import copy
import operator



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
        parent = self.parent
        while parent != None:
            path_points.append(parent.pIndex)
            parent = parent.parent
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
                    
                    print("Path:", str(newFNode)) 
                    #第一次最先拓展到终点的节点
                    if newFNode.state == 1 and possibleMinCostPoint == None:   
                        possibleMinCostPoint = newFNode  
                        
        #END while===========================================================================
        
        return getPathByNode(possibleMinCostPoint)
    
    else:
        return [goal]
    
    
    
    
#===================================================================
# PathItem Start
#===================================================================
#路径类
class PathItem(object):
    
    def __init__(self):
        #路径中的点列表
        self.path_points = []
        
        #路径中的点列表坐标字典
        self.path_points_dict = {}
        
        #当前路径的长度
        self.current_path_len = 0.0 

        #评估结果: path length + estimated distance
        self.evaluated_result = 0.0
        
        #路径末端的点
        self.frontier_index = 0
        
    def __str__(self):
        return "Points: %s, 评估结果值： %f" %(str(self.path_points), self.evaluated_result)
    
    def addPathPoint(self, point_index, pointXY, pointGoal):
        currentLen = len(self.path_points)
        #新加入节点到目标的预测距离
        estimated_distance = estimated_distance_2_goal(pointXY, pointGoal)
        if currentLen > 0:
            self.current_path_len += path_length_between_2_points(self.path_points_dict[self.path_points[currentLen - 1]], pointXY)
        
        self.evaluated_result = self.current_path_len + estimated_distance   

        self.path_points.append(point_index)
        self.frontier_index = point_index
        self.path_points_dict[point_index] = pointXY

    #拓展路径
    def expandPath(self, point_index, pointXY, pointGoal):
        expandedPath = copy.deepcopy(self)
        
        pointSize = len(expandedPath.path_points)
        #新加入节点到目标的预测距离
        estimated_distance = estimated_distance_2_goal(pointXY, pointGoal)
        if pointSize > 0:
            expandedPath.current_path_len += path_length_between_2_points(expandedPath.path_points_dict[expandedPath.path_points[pointSize - 1]], pointXY)
        expandedPath.evaluated_result = expandedPath.current_path_len + estimated_distance
  
        expandedPath.path_points.append(point_index)
        expandedPath.frontier_index = point_index
        expandedPath.path_points_dict[point_index] = pointXY
        return expandedPath
#===================================================================
# PathItem End
#===================================================================


    
def shortest_path1(M,start,goal):
    print("shortest path called")
    
    if(start != goal):
        #计算所有点到目标点的预测距离
        #estimated_distances_dict = estimated_distances_2_goal(M.intersections, M.intersections[goal])

        #查找过程中所有拓展路径字典
        expandedPathDict = {}

        #最短路径Index
        shortestPathIndex = 0

        #初始化起点
        startPath = PathItem()
        startPath.addPathPoint(start, M.intersections[start],M.intersections[goal])
        expandedPathDict[0] = startPath

        while True:
            #========拓展路径 START ========
            #获取预测结果最短的路径
            shortestPath = expandedPathDict[shortestPathIndex]

            #获取最短路径最后一个路径节点(边缘)的下标
            shortestFontierIndex = shortestPath.frontier_index

            roads = M.roads[shortestFontierIndex] #边缘节点可拓展的交叉点
            pathPointSet = set(shortestPath.path_points) #最短路径的所有节点

            count = 0
            for i in range(len(roads)):
                #1.不能拓展已经搜索过的交叉点(已经添加到路径中的节点); 2.排除闭环路径(不能再进行拓展的路径)  
                diffSet = pathPointSet.symmetric_difference(set(M.roads[roads[i]]))#路径边缘节点的下一个节点的roads的不重复的元素集合
                if not(roads[i] in pathPointSet) and not(len(diffSet) == 0):#要是diffSet等于0，则会形成闭环路径
                    expandedPath = shortestPath.expandPath(roads[i], M.intersections[roads[i]], M.intersections[goal])
                    if count == 0:
                        expandedPathDict[shortestPathIndex] = expandedPath
                    else:
                        expandedPathDict[len(expandedPathDict)] = expandedPath
                    count +=1
                    
            #========拓展路径 END ========
            
            #拓展路径后，重新确定f(path)最小的路径
            for key in expandedPathDict:
                shortestPath = expandedPathDict[shortestPathIndex]
                if expandedPathDict[key].evaluated_result <= shortestPath.evaluated_result:
                    shortestPathIndex = key

            #当路径已经拓展到目的地，且已经拓展到目的地的路径的评估结果(f(path) = path cost so far + estimated distance) 的值最小时,结束循环
            if(goal in set(expandedPathDict[shortestPathIndex].path_points)):
                break;

        return expandedPathDict[shortestPathIndex].path_points
    
    else:
        return [goal]
    
#Jiangchun
#chiangchuna@gmail.com
#May 25, 2019  20:52 
#May 26, 2019  22:16 

