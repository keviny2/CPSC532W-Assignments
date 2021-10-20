# https://www.geeksforgeeks.org/python-program-for-topological-sorting/
# This code is contributed by Neelam Yadav (modified)

# Python program to print topological sorting of a DAG
from collections import defaultdict


# Class to represent a graph
class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.V = vertices  # list of vertex names

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for vertex in self.graph[v]:
            if visited[vertex] == False:
                self.topologicalSortUtil(vertex, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        values = [False] * len(self.V)
        visited = dict(zip(self.V, values))
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for vertex in self.V:
            if visited[vertex] == False:
                self.topologicalSortUtil(vertex, visited, stack)

        # Print contents of stack
        return stack