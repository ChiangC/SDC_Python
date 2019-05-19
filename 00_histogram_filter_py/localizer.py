#import pdb
from helpers import normalize, blur

def initialize_beliefs(grid):
    height = len(grid)
    width = len(grid[0])
    area = height * width
    belief_per_cell = 1.0 / area
    beliefs = []
    for i in range(height):
        row = []
        for j in range(width):
            row.append(belief_per_cell)
        beliefs.append(row)
    return beliefs

def sense(color, grid, beliefs, p_hit, p_miss):
    new_beliefs = []

    #
    # TODO - implement this in part 2
    #
    height = len(grid)
    width = len(grid[0])
    for i in range(height):
        row = []
        for j in range(width):
            #pdb.set_trace()
            row.append((p_hit if (color == grid[i][j]) else p_miss)*beliefs[i][j])
        new_beliefs.append(row)
        
    p_sum = 0.    
    for m in range(len(new_beliefs)):    
        p_sum += sum(new_beliefs[m])
    
    for k in range(len(new_beliefs)):
        for h in range(len(new_beliefs[0])):
            new_beliefs[k][h] = new_beliefs[k][h]/p_sum
    #print(new_beliefs)        
    return new_beliefs

#Bug fixed
def move(dy, dx, beliefs, blurring):
    height = len(beliefs)
    width = len(beliefs[0])
    new_G = [[0.0 for i in range(width)] for j in range(height)]
    for i, row in enumerate(beliefs):
        for j, cell in enumerate(row):
            new_i = (i + dy) % height
            new_j = (j + dx) % width
            # pdb.set_trace()
            new_G[int(new_i)][int(new_j)] = cell
    return blur(new_G, blurring)

#Bug fixed with log print
# def move(dy, dx, beliefs, blurring):
#     height = len(beliefs)
#     width = len(beliefs[0])
#     print("localizer move 前======================================")
#     print("beliefs's rows:%d, cols:%d" %(len(beliefs), len(beliefs[0])))
#     new_G = [[0.0 for i in range(width)] for j in range(height)]
#     print("初始化后：new_G's rows:%d, cols:%d" %(len(new_G), len(new_G[0])))
#     for i, row in enumerate(beliefs):
#         for j, cell in enumerate(row):
#             new_i = (i + dy) % height
#             new_j = (j + dx) % width
#             # pdb.set_trace()
#             new_G[int(new_i)][int(new_j)] = cell
#     print("localizer move 后======================================")
#     print("new_G's rows:%d, cols:%s" %(len(new_G), len(new_G[0])))
#     return blur(new_G, blurring)

#function move with bug
# def move(dy, dx, beliefs, blurring):
#     height = len(beliefs)
#     width = len(beliefs[0])
#     new_G = [[0.0 for i in range(height)] for j in range(width)] #bug1
#     for i, row in enumerate(beliefs):
#         for j, cell in enumerate(row):
#             new_i = (i + dy) % height
#             new_j = (j + dx) % width
#             # pdb.set_trace()
#             new_G[int(new_j)][int(new_i)] = cell #bug2
#     return blur(new_G, blurring)



