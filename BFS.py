from queue import Queue
adj_list={
"M":["N","Q","R"],
"N":["M","O","Q"],
"O":["N","P"],
"P":["O","Q"],
"Q":["M","N","P"],
"R":["M"]
}
#initialization
visited= {} #keeping track of all the visited nodes
level={} #keeping track of level of each node
parent={}#keeping track of parent node of each node
bfs_traversal = [] #traversal list
queue = Queue()
for node in adj_list.keys():
    visited[node] = False
    parent[node] = None
    level[ node] = -1
print("Before traversal")
print("visited:",visited)
print("level:",level)
print("parent:",parent)

source = "M"
visited[source] = True
level[source] = 0 #SOURCE NONE IS m SO LEVEL WILL BE ZERO
queue.put(source) #ADD M TO THE QUEUE

while not queue.empty():
    u = queue.get() #GET THE FIRST ELEMENT FROM THE QUEUE
    bfs_traversal.append(u)
    for v in adj_list[u]:
        if not visited [v]:
            visited [v]= True
            parent[v]=u
            level[v]=level[u]+1
            queue.put(v)
print("After traversal")
print("BFS traversal:",bfs_traversal)
##Minimum Distance
print ("Minimum distance")
print("Level N",level["N"])
print("Level O",level["O"])
print("Parent M",parent["M"])
print("Parent P",parent["P"])
node = "O" #destination node
path= []
while node is not None:
    path.append(node)
    node = parent[node]
path.reverse()
print("Shortest path is:", path)
