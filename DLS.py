adj_list = {
    'A': ['B', 'C', 'D'],
    'B': ['E'],
    'C': ['F', 'G'],
    'D': ['C'],
    'E': ['H', 'F'],
    'F': ['B'],
    'H': [],
    'G': [],
}
visited = {}
parent = {}
dls_traversal = []

# Initialize visited and parent dictionaries
for node in adj_list.keys():
    visited[node] = False
    parent[node] = None

def dls(node, depth):
    if depth < 0:
        return False
    visited[node] = True
    dls_traversal.append(node)
    if node == target:
        return True
    for neighbor in adj_list[node]:
        if not visited[neighbor]:
            parent[neighbor] = node
            if dls(neighbor, depth - 1):
                return True
    return False

source = input("Enter the source node: ")
if source not in adj_list:
    print("Source node not in graph.")
else:
    depth_limit = int(input("Enter the depth limit: "))
    target = input("Enter the destination node: ")

    if target not in adj_list:
        print("Destination node not in graph.")
    else:
        found = dls(source, depth_limit)

        if not found:
            print("No path exists from source to destination within the depth limit.")
        else:
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = parent[node]

            path.reverse()
            print("DLS Traversal: ", dls_traversal)
            print("Path is: ", path)
