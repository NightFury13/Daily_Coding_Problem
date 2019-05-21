"""
This problem was asked by Google.

Given an undirected graph represented as an adjacency matrix and an integer k, 
write a function to determine whether each vertex in the graph can be colored 
such that no two adjacent vertices share the same color using at most k colors.
"""

class Graph:
    def __init__(self, n_vertices, graph):
        self.n_vertices = n_vertices
        self.graph = graph

    def check_color(self, v_id, colors, cur_color):
        for i in range(self.n_vertices):
            if self.graph[v_id][i] == 1 and colors[i] == cur_color:
                return False
        return True

    def fill_util(self, colors, k, n_filled):
        # All filled!
        if n_filled == self.n_vertices:
            return True
        
        for col in range(1, k+1):
            if not self.check_color(n_filled, colors, col):
                continue

            colors[n_filled] = col
            if self.fill_util(colors, k, n_filled+1):
                return True
            colors[n_filled] = 0

        return False

    def fill_color(self, k):
        colors = [0 for i in range(self.n_vertices)]

        for vert in range(self.n_vertices):
            if not self.fill_util(colors, k, vert):
                return False
        
        return colors

if __name__ == '__main__':
    matrix = [[0, 1, 1, 1],
              [1, 0, 1, 0],
              [1, 1, 0, 0],
              [1, 0, 0, 0]]
    graph = Graph(4, matrix)
    print(matrix)

    k = 3
    print(k, graph.fill_color(k))
    k = 2
    print(k, graph.fill_color(k))
