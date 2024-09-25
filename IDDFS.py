graph = {
     "M" : ["N","Q","R"],
     "N" : ["O","Q","M"],
     "R" : ["M"],
     "O" : ["P","N"],
     "Q" : ["M","N"],
     "P" : ["O","Q"]
     }
def dls(graph,S,D,L):
        visited = {}
        parent = {}
        level = {}
        dls_list = []
        shortest_path = []
        stack = []
        for i in graph.keys():
            visited[i] = False
            parent[i] = None
            level[i] = -1
        stack.append(S)
        level[S] = 0
        while stack:
            u = stack.pop()
            visited[u] = True
            dls_list.append(u)
            for v in graph[u]:
                if not visited[v] and v not in stack and level[u] + 1<=L:
                    level[v] = level[u] + 1
                    parent[v] = u
                    stack.append(v)
            if D in dls_list:
                temp = D
                while parent[temp] != None:
                    shortest_path.append(temp)
                    temp = parent[temp]
                else:
                    print(f"{D} found in level {L}")
                    shortest_path.append(S)
                    shortest_path.reverse()
                    print(f"Shortest path is: {shortest_path}")
            else:
                print(f"{D} not found in level {L}")
            print(f"DFS till level {L}: {dls_list}\n\n")
S = input("Enter the source node: ")
D = input("Enter the destination node: ")
level = int(input("Enter the limit level: "))
for i in range(level):
    dls(graph,S,D,i)
