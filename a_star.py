graph = {
    "Oradea": ({"Zerind": 71, "Sibiu": 151}, 380),
    "Zerind": ({"Oradea": 71, "Arad": 75}, 374),
    "Arad": ({"Zerind": 75, "Sibiu": 140 , "Timisoara": 118}, 399),
    "Timisoara": ({"Arad": 118, "Lugoj": 111}, 329),
    "Lugoj": ({"Timisoara": 111, "Mehadia": 70}, 244),
    "Mehadia": ({"Lugoj": 70, "Drobeta": 75}, 241),
    "Sibiu": ({"Oradea": 151, "Fagaras": 99, "Rimnicu Vilcea": 80, "Arad": 140}, 253),
    "Fagaras": ({"Sibiu": 99, "Bucharest": 211}, 176),
    "Rimnicu Vilcea": ({"Sibiu": 80, "Pitesti": 97, "Craiova": 146}, 193),
    "Bucharest": ({"Fagaras": 211, "Pitesti": 101, "Urziceni": 85, "Giurgiu": 90}, 0),
    "Pitesti": ({"Rimnicu Vilcea": 97, "Craiova": 138, "Bucharest": 101}, 100),
    "Craiova": ({"Rimnicu Vilcea": 146, "Pitesti": 138, "Drobeta": 120}, 160),
    "Drobeta": ({"Mehadia": 75, "Craiova": 120}, 242),
    "Urziceni": ({"Bucharest": 85, "Vaslui": 142, "Hirsova": 98}, 80),
    "Giurgiu": ({"Bucharest": 90}, 77),
    "Vaslui": ({"Iasi": 92, "Urziceni": 142}, 199),
    "Hirsova": ({"Urziceni": 98, "Eforie": 86}, 151),
    "Iasi": ({"Vaslui": 92, "Neamt": 87}, 226),
    "Eforie": ({"Hirsova": 86}, 161),
    "Neamt": ({"Iasi": 87}, 234)
    }

def get_min(q):
    mn = (0, (0, float("INF")))
    for i in q:
        if sum(q[i]) < sum(mn[1]):
            mn = (i, q[i])
    return mn[0]
    
def a_star(graph, prev, dst, path, pcost, q):
    print("Connected nodes of current node", prev, "with h(n) values: ")    
    for n in graph[prev][0]:    #neighbors list n =Z, S, T
        if n not in path:
            q[n] = (graph[n][1], graph[prev][0][n])
            print(n, "->", q[n])
            add1=sum(q[n])
            path_cost=pcost+add1        
            print("A* value for ",n, "is: ",path_cost)
    while q:
        mn = get_min(q)
        print("Selecting Minimun vertex: ", mn)
        print("__________________________________________________")
        if dst == mn:
            return path + [dst]
        pc = pcost + q[mn][1]
        print("Previous path cost:",pc)
        #del q[mn]
        new_path = a_star(graph, mn, dst, path + [mn], pc, q)
        if new_path:
            return new_path
    return []
source=input("Enter Source vertex: ")
dest=input("Enter destination vertex: ")
heuristic=int(input("Enter given heuristic value for source: "))
path = a_star(graph, source, dest, [], 0, {source: (heuristic, 0)})
if path:
    print(path)
else:
    print("Path not found")

