import math
import copy

#f = path length + estimated distance

#计算两点之间的距离(path cost)
#param point1,point2: 是包含x,y坐标的列表，类似[0.7798606835438107, 0.6922727646627362]
#return: 返回两点之间的距离
def path_length_between_2_points(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

#所有的点到目标点的预测距离(estimated distance to goal)
#param intersections: 字典；类似{0: [0.7798606835438107, 0.6922727646627362],1: [0.7647837074641568, 0.3252670836724646]}
#param goal: 是包含x,y坐标的列表，类似[0.7798606835438107, 0.6922727646627362]
#return: 返回一个字典，类似{0: p0_distance_2_goal ,1: p1_distance_2_goal}
def estimated_distances_2_goal(intersections, goal):
    estimated_distances_dict = {}
    for key in intersections:
        estimated_distances_dict[key] = path_length_between_2_points(intersections[key], goal)
    return estimated_distances_dict

def estimated_distance_2_goal(point, pGoal):
    return path_length_between_2_points(point, pGoal)

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

    
def shortest_path(M,start,goal):
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

