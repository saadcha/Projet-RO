import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog
import tkinter.messagebox as messagebox
import networkx as nx
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import string
import numpy as np  # Import NumPy for matrix operations
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # Dictionary to store the graph
        self.V = vertices                 # Number of vertices in the graph
        self.capacity = {}                # Dictionary to store capacities

    def add_edge(self, u, v, capacity):
        """Adds an edge (u -> v) with a given capacity."""
        self.graph[u].append(v)
        self.graph[v].append(u)  # Add the reverse edge for the residual graph
        self.capacity[(u, v)] = capacity
        self.capacity[(v, u)] = 0  # Initial reverse capacity (for the residual graph)

    def _dfs(self, source, visited):
        """Depth-first search (DFS) to mark reachable vertices from the source."""
        visited[source] = True
        for neighbor in self.graph[source]:
            if not visited[neighbor] and self.capacity[(source, neighbor)] > 0:
                self._dfs(neighbor, visited)

    def ford_fulkerson(self, source, sink):
        """Implements the Ford-Fulkerson algorithm to calculate the maximum flow."""
        parent = [-1] * self.V
        max_flow = 0

        while True:
            visited = [False] * self.V
            if not self._dfs_find_path(source, sink, visited, parent):
                break

            # Find the minimum capacity in the augmenting path
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.capacity[(parent[s], s)])
                s = parent[s]

            # Update residual capacities
            v = sink
            while v != source:
                u = parent[v]
                self.capacity[(u, v)] -= path_flow
                self.capacity[(v, u)] += path_flow
                v = parent[v]

            max_flow += path_flow

        return max_flow

    def _dfs_find_path(self, source, sink, visited, parent):
        """Finds an augmenting path using DFS."""
        visited[source] = True
        if source == sink:
            return True
        for neighbor in self.graph[source]:
            if not visited[neighbor] and self.capacity[(source, neighbor)] > 0:
                parent[neighbor] = source
                if self._dfs_find_path(neighbor, sink, visited, parent):
                    return True
        return False

    def min_cut(self, source):
        """Finds and displays the edges of the minimum cut after running Ford-Fulkerson."""
        visited = [False] * self.V
        self._dfs(source, visited)

        min_cut_edges = []
        for u in range(self.V):
            for v in self.graph[u]:
                if visited[u] and not visited[v] and self.capacity[(u, v)] == 0:
                    min_cut_edges.append((u, v))

        return min_cut_edges

class OperationsResearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Operations Research GUI")
        self.root.geometry("800x600")
        
        # Main title
        title_label = tk.Label(
            root, 
            text="Operations Research GUI",
            font=("Arial", 16, "bold"),
            pady=10
        )
        title_label.pack()

        # Main buttons frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # Main navigation buttons
        tk.Button(
            button_frame, 
            text="Input",
            command=self.show_input_screen,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame, 
            text="Output",
            command=self.show_output_screen,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame, 
            text="Operations Research Algorithm",
            command=self.show_algorithms,
            width=30
        ).pack(side=tk.LEFT, padx=5)

        # Algorithms frame
        self.algorithms_frame = tk.Frame(root)
        self.algorithms_frame.pack(pady=20, fill=tk.BOTH, expand=True)

        # Algorithm buttons
        self.create_algorithm_buttons()

        # Code display area
        self.code_display = scrolledtext.ScrolledText(
            root,
            width=70,
            height=20,
            font=("Courier", 10)
        )
        self.code_display.pack(pady=10, padx=10)

        # Exit button
        tk.Button(
            root,
            text="Exit",
            command=self.exit_application,
            width=20
        ).pack(pady=10)

    def create_algorithm_buttons(self):
        algorithms = [
            ("Welsh Powell", self.run_welsh_powell),
            ("Dijkstra", self.run_dijkstra),
            ("Kruskal", self.run_kruskal),
            ("Moindre Coût", self.moindre_cout_callback),
            ("Nord-Ouest", self.nord_ouest_callback),
            ("Stepping-Stone", self.stepping_stone_callback),
            ("Bellman Ford", self.bellman_ford_callback),
            ("Ford Fulkerson", self.ford_fulkerson_callback),
            "Potentiel Metra"
        ]

        for i, algo in enumerate(algorithms):
            if isinstance(algo, tuple):
                name, command = algo
                btn = tk.Button(
                    self.algorithms_frame,
                    text=name,
                    command=command,
                    width=20
                )
            else:
                btn = tk.Button(
                    self.algorithms_frame,
                    text=algo,
                    command=lambda x=algo: self.show_algorithm_code(x),
                    width=20
                )
            row = i // 3
            col = i % 3
            btn.grid(row=row, column=col, padx=5, pady=5)

    def show_algorithm_code(self, algorithm):
        if algorithm == "Dijkstra":
            self.run_dijkstra()
            return
        
        # Dictionary containing sample code for each algorithm
        algorithm_codes = {
            "Welsh Powell": """def welsh_powell(graph):
    # Algorithm implementation
    pass""",
            "Dijkstra": """def dijkstra(graph, start):
    # Algorithm implementation
    pass""",
            "Kruskal": """def kruskal(graph):
    # Algorithm implementation
    pass""",
            # Add other algorithms here
        }

        code = algorithm_codes.get(algorithm, "Code not available")
        self.code_display.delete(1.0, tk.END)
        self.code_display.insert(tk.END, f"Algorithm: {algorithm}\n\n{code}")

    def show_input_screen(self):
        messagebox.showinfo("Input", "Input screen functionality")

    def show_output_screen(self):
        messagebox.showinfo("Output", "Output screen functionality")

    def show_algorithms(self):
        messagebox.showinfo("Algorithms", "Showing available algorithms")

    def exit_application(self):
        if messagebox.askokcancel("Exit", "Do you want to exit the application?"):
            self.root.destroy()

    def print_graph_to_console(self, G):
        print(f"Nombre de sommets : {G.number_of_nodes()}")
        print(f"Nombre d'arêtes : {G.number_of_edges()}")
        print("Liste des arêtes :")
        for edge in G.edges():
            print(str(edge))  # Ensure edges are printed as strings

    def run_dijkstra(self):
        try:
            # Clear previous content
            self.code_display.delete(1.0, tk.END)
            
            # Get number of vertices
            n = simpledialog.askinteger("Input", 
                "Entrez le nombre de sommets pour le graphe :", 
                parent=self.root,
                minvalue=8)  # Adjusted for 8 nodes
            
            if not n:  # User cancelled
                return
            
            # Create graph
            G = nx.Graph()
            nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # Define your nodes
            for node in nodes:
                G.add_node(node)
            
            # Add edges with weights (as per your image)
            edges = [
                ('A', 'B', 2),
                ('A', 'C', 3),
                ('A', 'D', 3),
                ('B', 'D', 3),
                ('B', 'F', 4),
                ('C', 'D', 2),
                ('C', 'E', 4),
                ('D', 'E', 5),
                ('D', 'F', 3),
                ('E', 'F', 2),
                ('E', 'G', 5),
                ('F', 'G', 3),
            ]
            G.add_weighted_edges_from(edges)
            
            # Get start and end nodes
            depart = simpledialog.askstring("Input", 
                f"Entrez le sommet de départ (A, B, C, D, E, F, G, H) :", 
                parent=self.root)
            
            if not depart:  # User cancelled
                return
            
            arrivee = simpledialog.askstring("Input", 
                f"Entrez le sommet d'arrivée (A, B, C, D, E, F, G, H) :", 
                parent=self.root)
            
            if not arrivee:  # User cancelled
                return
            
            # Add console visualization of initial graph
            self.code_display.insert(tk.END, "État initial du graphe:\n")
            self.print_graph_to_console(G)
            
            # Calculate shortest path
            try:
                chemin = nx.dijkstra_path(G, depart, arrivee, weight='weight')
                distance_totale = nx.dijkstra_path_length(G, depart, arrivee, weight='weight')
                
                # Display results
                self.code_display.insert(tk.END, f"\nRésultats de l'algorithme de Dijkstra:\n")
                self.code_display.insert(tk.END, f"Le plus court chemin entre {depart} et {arrivee} est : {' -> '.join(chemin)}\n")
                self.code_display.insert(tk.END, f"Distance totale : {distance_totale}\n\n")
                
                # Create visualization
                plt.clf()  # Clear any existing plots
                fig = plt.figure(figsize=(8, 8))
                
                def create_custom_layout(G):
                    # Define positions for each node based on the provided graph layout
                    pos = {
                        'A': (0, 2),  # Position for A
                        'B': (1, 3),  # Position for B
                        'C': (0, 1),  # Position for C
                        'D': (1, 2),  # Position for D
                        'E': (2, 1),  # Position for E
                        'F': (2, 3),  # Position for F
                        'G': (3, 2),  # Position for G
                    }

                    # Ensure all nodes in the graph have positions (fallback for extra nodes)
                    for node in G.nodes():
                        if node not in pos:
                            pos[node] = (random.random(), random.random())

                    return pos

                pos = create_custom_layout(G)
                edges = G.edges(data=True)
                
                # Draw edges with colors indicating the shortest path
                edge_colors = [
                    'red' if (u, v) in zip(chemin, chemin[1:]) or (v, u) in zip(chemin, chemin[1:]) else 'black'
                    for u, v, d in G.edges(data=True)
                ]
                
                # Draw nodes with specific colors for those in the shortest path
                node_colors = ['green' if node in chemin else 'lightblue' for node in G.nodes()]
                
                # Draw the graph
                nx.draw(
                    G, pos,
                    with_labels=True,
                    node_color=node_colors,
                    edge_color=edge_colors,
                    node_size=1000,
                    font_size=16,
                    font_weight='bold',
                    width=2
                )
                
                # Add edge labels (weights)
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=14)
                
                # Adjust plot layout
                plt.axis('equal')  # Make the plot aspect ratio 1:1
                plt.margins(0.2)  # Add some margins around the graph
                
                # Display plot in GUI
                if hasattr(self, 'canvas'):
                    self.canvas.get_tk_widget().destroy()
                self.canvas = FigureCanvasTkAgg(fig, master=self.root)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                
            except nx.NetworkXNoPath:
                messagebox.showerror("Error", 
                    f"Aucun chemin entre {depart} et {arrivee}.")
            except Exception as e:
                messagebox.showerror("Error", 
                    f"Une erreur s'est produite: {str(e)}")
            
        except Exception as e:
            messagebox.showerror("Error", 
                f"Une erreur s'est produite: {str(e)}")

    def run_welsh_powell(self):
        try:
            # Prompt user for number of vertices
            n = simpledialog.askinteger("Input", "Entrez le nombre de sommets pour le graphe :", parent=self.root, minvalue=1)
            if n is None:  # User cancelled
                return
            
            # Generate a random graph
            G = nx.Graph()
            G.add_nodes_from(range(n))  # Add n nodes
            
            # Add edges with 50% probability
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() > 0.5:
                        G.add_edge(i, j)

            self.code_display.insert(tk.END, f"Graph généré avec {n} sommets et {G.number_of_edges()} arêtes.\n")
            self.print_graph_to_console(G)  # Display edges in console
            
            # Welsh Powell algorithm
            colors = self.welsh_powell_coloring(G)
            self.visualize_graph(G, colors)

        except Exception as e:
            messagebox.showerror("Error", f"Une erreur s'est produite: {str(e)}")

    def welsh_powell_coloring(self, G):
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        node_colors = {}
        for node in sorted_nodes:
            neighbor_colors = {node_colors[neighbor] for neighbor in G.neighbors(node) if neighbor in node_colors}
            color = 0
            while color in neighbor_colors:
                color += 1
            node_colors[node] = color
        return node_colors

    def visualize_graph(self, G, colors):
        color_map = [colors[node] for node in G.nodes()]
        
        plt.clf()  # Clear any existing plots
        nx.draw(G, with_labels=True, node_color=color_map, edge_color='gray', node_size=700, font_size=10)
        plt.title("Welsh Powell Graph Coloring")
        plt.show()

    def run_kruskal(self):
        try:
            n = simpledialog.askinteger("Input", "Entrez le nombre de sommets pour le graphe :", parent=self.root, minvalue=2)
            if n is None:  # User cancelled
                return
            
            G = self.generer_graphe_aleatoire(n)
            mst, cout_mst = self.calculer_mst_kruskal(G)

            self.code_display.insert(tk.END, f"Le coût de l'arbre de minimum spanning tree : {cout_mst}\n")
            self.print_graph_to_console(G)  # Display edges in console
            self.dessiner_graphe_et_mst(G, mst)

        except Exception as e:
            messagebox.showerror("Error", f"Une erreur s'est produite: {str(e)}")

    def generer_noms_sommets(self, n):
        alphabet = list(string.ascii_uppercase)
        if n <= 26:
            return alphabet[:n]
        else:
            noms_double_lettre = [a + b for a in alphabet for b in alphabet]
            return alphabet + noms_double_lettre[:n - 26]

    def generer_graphe_aleatoire(self, n):
        G = nx.Graph()
        sommets = self.generer_noms_sommets(n)
        for i in range(n):
            for j in range(i + 1, n):
                poids = random.randint(1, 100)
                G.add_edge(sommets[i], sommets[j], weight=poids)
        return G

    def calculer_mst_kruskal(self, G):
        mst = nx.Graph()
        arretes_tries = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
        parent = {i: i for i in G.nodes()}

        def trouver_parent(noeud):
            if parent[noeud] == noeud:
                return noeud
            return trouver_parent(parent[noeud])

        def fusionner_ensembles(u, v):
            racine_u = trouver_parent(u)
            racine_v = trouver_parent(v)
            parent[racine_u] = racine_v

        cout_mst = 0
        for u, v, data in arretes_tries:
            if trouver_parent(u) != trouver_parent(v):
                mst.add_edge(u, v, weight=data['weight'])
                cout_mst += data['weight']
                fusionner_ensembles(u, v)

        return mst, cout_mst

    def dessiner_graphe_et_mst(self, G, mst):
        pos = nx.spring_layout(G)
        etiquettes_arretes = nx.get_edge_attributes(G, 'weight')

        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=etiquettes_arretes)

        arretes_mst = list(mst.edges())
        nx.draw_networkx_edges(G, pos, edgelist=arretes_mst, edge_color='r', width=2)

import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog
import tkinter.messagebox as messagebox
import networkx as nx
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import string
import numpy as np  # Import NumPy for matrix operations
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # Dictionary to store the graph
        self.V = vertices                 # Number of vertices in the graph
        self.capacity = {}                # Dictionary to store capacities

    def add_edge(self, u, v, capacity):
        """Adds an edge (u -> v) with a given capacity."""
        self.graph[u].append(v)
        self.graph[v].append(u)  # Add the reverse edge for the residual graph
        self.capacity[(u, v)] = capacity
        self.capacity[(v, u)] = 0  # Initial reverse capacity (for the residual graph)

    def _dfs(self, source, visited):
        """Depth-first search (DFS) to mark reachable vertices from the source."""
        visited[source] = True
        for neighbor in self.graph[source]:
            if not visited[neighbor] and self.capacity[(source, neighbor)] > 0:
                self._dfs(neighbor, visited)

    def ford_fulkerson(self, source, sink):
        """Implements the Ford-Fulkerson algorithm to calculate the maximum flow."""
        parent = [-1] * self.V
        max_flow = 0

        while True:
            visited = [False] * self.V
            if not self._dfs_find_path(source, sink, visited, parent):
                break

            # Find the minimum capacity in the augmenting path
            path_flow = float('inf')
            s = sink
            while s != source:
                path_flow = min(path_flow, self.capacity[(parent[s], s)])
                s = parent[s]

            # Update residual capacities
            v = sink
            while v != source:
                u = parent[v]
                self.capacity[(u, v)] -= path_flow
                self.capacity[(v, u)] += path_flow
                v = parent[v]

            max_flow += path_flow

        return max_flow

    def _dfs_find_path(self, source, sink, visited, parent):
        """Finds an augmenting path using DFS."""
        visited[source] = True
        if source == sink:
            return True
        for neighbor in self.graph[source]:
            if not visited[neighbor] and self.capacity[(source, neighbor)] > 0:
                parent[neighbor] = source
                if self._dfs_find_path(neighbor, sink, visited, parent):
                    return True
        return False

    def min_cut(self, source):
        """Finds and displays the edges of the minimum cut after running Ford-Fulkerson."""
        visited = [False] * self.V
        self._dfs(source, visited)

        min_cut_edges = []
        for u in range(self.V):
            for v in self.graph[u]:
                if visited[u] and not visited[v] and self.capacity[(u, v)] == 0:
                    min_cut_edges.append((u, v))

        return min_cut_edges

class OperationsResearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interface Graphique Tkinter GUI")
        self.root.geometry("800x600")
        # Changer la couleur de fond de la fenêtre principale en rouge
        self.root.configure(bg='red')
        
        # Main title
        title_label = tk.Label(
            root, 
            text="Interface Graphique Tkinter GUI",
            font=("Arial", 16, "bold"),
            pady=10,
             bg='red',  # Fond du label en rouge
            fg='white'  # Texte en blanc
        )
        title_label.pack()

        # Main buttons frame
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        # Main navigation buttons
        tk.Button(
            button_frame, 
            text="Entrée",
            command=self.show_input_screen,
            width=20,
             
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame, 
            text="Sortie",
            command=self.show_output_screen,
            width=20,
              
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            button_frame, 
            text="Algorithme de Recherche Opérationnelle",
            command=self.show_algorithms,
            width=30,
             
        ).pack(side=tk.LEFT, padx=5)

         # Create a central frame for the algorithm buttons
        self.algorithms_frame = tk.Frame(root)
        self.algorithms_frame.pack(pady=20)  # Centrer le cadre

        # Algorithm buttons
        self.create_algorithm_buttons()

        # Code display area
        self.code_display = scrolledtext.ScrolledText(
            root,
            width=70,
            height=20,
            font=("Courier", 10),
             
        )
        self.code_display.pack(pady=10, padx=10)

        # Exit button
        tk.Button(
            root,
            text="Exit",
            command=self.exit_application,
            width=20,
             
        ).pack(pady=10)

    def create_algorithm_buttons(self):
        algorithms = [
            ("Welsh Powell", self.run_welsh_powell),
            ("Dijkstra", self.run_dijkstra),
            ("Kruskal", self.run_kruskal),
            ("Moindre Coût", self.moindre_cout_callback),
            ("Nord-Ouest", self.nord_ouest_callback),
            ("Stepping-Stone", self.stepping_stone_callback),
            ("Bellman Ford", self.bellman_ford_callback),
            ("Ford Fulkerson", self.ford_fulkerson_callback),
            "Potentiel Metra"
        ]

        for i, algo in enumerate(algorithms):
            if isinstance(algo, tuple):
                name, command = algo
                btn = tk.Button(
                    self.algorithms_frame,
                    text=name,
                    command=command,
                    width=20,
                     
                )
            else:
                btn = tk.Button(
                    self.algorithms_frame,
                    text=algo,
                    command=lambda x=algo: self.show_algorithm_code(x),
                    width=20,
                     
                )
            row = i // 3
            col = i % 3
            btn.grid(row=row, column=col, padx=5, pady=5)

    def show_algorithm_code(self, algorithm):
        if algorithm == "Dijkstra":
            self.run_dijkstra()
            return
        
        # Dictionary containing sample code for each algorithm
        algorithm_codes = {
            "Welsh Powell": """def welsh_powell(graph):
    # Algorithm implementation
    pass""",
            "Dijkstra": """def dijkstra(graph, start):
    # Algorithm implementation
    pass""",
            "Kruskal": """def kruskal(graph):
    # Algorithm implementation
    pass""",
            # Add other algorithms here
        }

        code = algorithm_codes.get(algorithm, "Code not available")
        self.code_display.delete(1.0, tk.END)
        self.code_display.insert(tk.END, f"Algorithm: {algorithm}\n\n{code}")

    def show_input_screen(self):
        messagebox.showinfo("Input", "Input screen functionality")

    def show_output_screen(self):
        messagebox.showinfo("Output", "Output screen functionality")

    def show_algorithms(self):
        messagebox.showinfo("Algorithms", "Showing available algorithms")

    def exit_application(self):
        if messagebox.askokcancel("Exit", "Do you want to exit the application?"):
            self.root.destroy()

    def print_graph_to_console(self, G):
        print(f"Nombre de sommets : {G.number_of_nodes()}")
        print(f"Nombre d'arêtes : {G.number_of_edges()}")
        print("Liste des arêtes :")
        for edge in G.edges():
            print(str(edge))  # Ensure edges are printed as strings

    def run_dijkstra(self):
        try:
            # Clear previous content
            self.code_display.delete(1.0, tk.END)
            
            # Get number of vertices
            n = simpledialog.askinteger("Input", 
                "Entrez le nombre de sommets pour le graphe :", 
                parent=self.root,
                minvalue=8)  # Adjusted for 8 nodes
            
            if not n:  # User cancelled
                return
            
            # Create graph
            G = nx.Graph()
            nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']  # Define your nodes
            for node in nodes:
                G.add_node(node)
            
            # Add edges with weights (as per your image)
            edges = [
                ('A', 'B', 2),
                ('A', 'C', 3),
                ('A', 'D', 3),
                ('B', 'D', 3),
                ('B', 'F', 4),
                ('C', 'D', 2),
                ('C', 'E', 4),
                ('D', 'E', 5),
                ('D', 'F', 3),
                ('E', 'F', 2),
                ('E', 'G', 5),
                ('F', 'G', 3),
            ]
            G.add_weighted_edges_from(edges)
            
            # Get start and end nodes
            depart = simpledialog.askstring("Input", 
                f"Entrez le sommet de départ (A, B, C, D, E, F, G, H) :", 
                parent=self.root)
            
            if not depart:  # User cancelled
                return
            
            arrivee = simpledialog.askstring("Input", 
                f"Entrez le sommet d'arrivée (A, B, C, D, E, F, G, H) :", 
                parent=self.root)
            
            if not arrivee:  # User cancelled
                return
            
            # Add console visualization of initial graph
            self.code_display.insert(tk.END, "État initial du graphe:\n")
            self.print_graph_to_console(G)
            
            # Calculate shortest path
            try:
                chemin = nx.dijkstra_path(G, depart, arrivee, weight='weight')
                distance_totale = nx.dijkstra_path_length(G, depart, arrivee, weight='weight')
                
                # Display results
                self.code_display.insert(tk.END, f"\nRésultats de l'algorithme de Dijkstra:\n")
                self.code_display.insert(tk.END, f"Le plus court chemin entre {depart} et {arrivee} est : {' -> '.join(chemin)}\n")
                self.code_display.insert(tk.END, f"Distance totale : {distance_totale}\n\n")
                
                # Create visualization
                plt.clf()  # Clear any existing plots
                fig = plt.figure(figsize=(8, 8))
                
                def create_custom_layout(G):
                    # Define positions for each node based on the provided graph layout
                    pos = {
                        'A': (0, 2),  # Position for A
                        'B': (1, 3),  # Position for B
                        'C': (0, 1),  # Position for C
                        'D': (1, 2),  # Position for D
                        'E': (2, 1),  # Position for E
                        'F': (2, 3),  # Position for F
                        'G': (3, 2),  # Position for G
                    }

                    # Ensure all nodes in the graph have positions (fallback for extra nodes)
                    for node in G.nodes():
                        if node not in pos:
                            pos[node] = (random.random(), random.random())

                    return pos

                pos = create_custom_layout(G)
                edges = G.edges(data=True)
                
                # Draw edges with colors indicating the shortest path
                edge_colors = [
                    'red' if (u, v) in zip(chemin, chemin[1:]) or (v, u) in zip(chemin, chemin[1:]) else 'black'
                    for u, v, d in G.edges(data=True)
                ]
                
                # Draw nodes with specific colors for those in the shortest path
                node_colors = ['green' if node in chemin else 'lightblue' for node in G.nodes()]
                
                # Draw the graph
                nx.draw(
                    G, pos,
                    with_labels=True,
                    node_color=node_colors,
                    edge_color=edge_colors,
                    node_size=1000,
                    font_size=16,
                    font_weight='bold',
                    width=2
                )
                
                # Add edge labels (weights)
                edge_labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=14)
                
                # Adjust plot layout
                plt.axis('equal')  # Make the plot aspect ratio 1:1
                plt.margins(0.2)  # Add some margins around the graph
                
                # Display plot in GUI
                if hasattr(self, 'canvas'):
                    self.canvas.get_tk_widget().destroy()
                self.canvas = FigureCanvasTkAgg(fig, master=self.root)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                
            except nx.NetworkXNoPath:
                messagebox.showerror("Error", 
                    f"Aucun chemin entre {depart} et {arrivee}.")
            except Exception as e:
                messagebox.showerror("Error", 
                    f"Une erreur s'est produite: {str(e)}")
            
        except Exception as e:
            messagebox.showerror("Error", 
                f"Une erreur s'est produite: {str(e)}")

    def run_welsh_powell(self):
        try:
            # Prompt user for number of vertices
            n = simpledialog.askinteger("Input", "Entrez le nombre de sommets pour le graphe :", parent=self.root, minvalue=1)
            if n is None:  # User cancelled
                return
            
            # Generate a random graph
            G = nx.Graph()
            G.add_nodes_from(range(n))  # Add n nodes
            
            # Add edges with 50% probability
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() > 0.5:
                        G.add_edge(i, j)

            self.code_display.insert(tk.END, f"Graph généré avec {n} sommets et {G.number_of_edges()} arêtes.\n")
            self.print_graph_to_console(G)  # Display edges in console
            
            # Welsh Powell algorithm
            colors = self.welsh_powell_coloring(G)
            self.visualize_graph(G, colors)

        except Exception as e:
            messagebox.showerror("Error", f"Une erreur s'est produite: {str(e)}")

    def welsh_powell_coloring(self, G):
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        node_colors = {}
        for node in sorted_nodes:
            neighbor_colors = {node_colors[neighbor] for neighbor in G.neighbors(node) if neighbor in node_colors}
            color = 0
            while color in neighbor_colors:
                color += 1
            node_colors[node] = color
        return node_colors

    def visualize_graph(self, G, colors):
        color_map = [colors[node] for node in G.nodes()]
        
        plt.clf()  # Clear any existing plots
        nx.draw(G, with_labels=True, node_color=color_map, edge_color='gray', node_size=700, font_size=10)
        plt.title("Welsh Powell Graph Coloring")
        plt.show()

    def run_kruskal(self):
        try:
            n = simpledialog.askinteger("Input", "Entrez le nombre de sommets pour le graphe :", parent=self.root, minvalue=2)
            if n is None:  # User cancelled
                return
            
            G = self.generer_graphe_aleatoire(n)
            mst, cout_mst = self.calculer_mst_kruskal(G)

            self.code_display.insert(tk.END, f"Le coût de l'arbre de minimum spanning tree : {cout_mst}\n")
            self.print_graph_to_console(G)  # Display edges in console
            self.dessiner_graphe_et_mst(G, mst)

        except Exception as e:
            messagebox.showerror("Error", f"Une erreur s'est produite: {str(e)}")

    def generer_noms_sommets(self, n):
        alphabet = list(string.ascii_uppercase)
        if n <= 26:
            return alphabet[:n]
        else:
            noms_double_lettre = [a + b for a in alphabet for b in alphabet]
            return alphabet + noms_double_lettre[:n - 26]

    def generer_graphe_aleatoire(self, n):
        G = nx.Graph()
        sommets = self.generer_noms_sommets(n)
        for i in range(n):
            for j in range(i + 1, n):
                poids = random.randint(1, 100)
                G.add_edge(sommets[i], sommets[j], weight=poids)
        return G

    def calculer_mst_kruskal(self, G):
        mst = nx.Graph()
        arretes_tries = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
        parent = {i: i for i in G.nodes()}

        def trouver_parent(noeud):
            if parent[noeud] == noeud:
                return noeud
            return trouver_parent(parent[noeud])

        def fusionner_ensembles(u, v):
            racine_u = trouver_parent(u)
            racine_v = trouver_parent(v)
            parent[racine_u] = racine_v

        cout_mst = 0
        for u, v, data in arretes_tries:
            if trouver_parent(u) != trouver_parent(v):
                mst.add_edge(u, v, weight=data['weight'])
                cout_mst += data['weight']
                fusionner_ensembles(u, v)

        return mst, cout_mst

    def dessiner_graphe_et_mst(self, G, mst):
        pos = nx.spring_layout(G)
        etiquettes_arretes = nx.get_edge_attributes(G, 'weight')

        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=etiquettes_arretes)

        arretes_mst = list(mst.edges())
        nx.draw_networkx_edges(G, pos, edgelist=arretes_mst, edge_color='r', width=2)

        plt.title("Graphe et Arbre de Minimum Spanning Tree")
        plt.show()

    def generate_random_data(self, num_magazines, num_usines):
        # Generate random transportation costs, factory capacities, and store demands
        couts_transport = np.random.randint(1, 100, size=(num_magazines, num_usines)).tolist()
        capacites_usines = np.random.randint(50, 200, size=num_usines).tolist()
        demandes_magazines = np.random.randint(30, 100, size=num_magazines).tolist()
        return couts_transport, capacites_usines, demandes_magazines

    def algorithme_moindre_cout(self, capacites_usines, demandes_magazines, couts_transport):
        # Implement the "Moindre Coût" algorithm here
        # This is a placeholder for the actual implementation
        allocation = np.zeros((len(demandes_magazines), len(capacites_usines))).tolist()
        # Example allocation logic (to be replaced with actual algorithm)
        for i in range(len(demandes_magazines)):
            for j in range(len(capacites_usines)):
                allocation[i][j] = min(demandes_magazines[i], capacites_usines[j])
                demandes_magazines[i] -= allocation[i][j]
                capacites_usines[j] -= allocation[i][j]
        return allocation

    def algorithme_nord_ouest(self, capacites_usines, demandes_magazines, couts_transport):
        # Implement the "Nord-Ouest" algorithm here
        # This is a placeholder for the actual implementation
        allocation = np.zeros((len(demandes_magazines), len(capacites_usines))).tolist()
        # Example allocation logic (to be replaced with actual algorithm)
        for i in range(len(demandes_magazines)):
            for j in range(len(capacites_usines)):
                allocation[i][j] = min(demandes_magazines[i], capacites_usines[j])
                demandes_magazines[i] -= allocation[i][j]
                capacites_usines[j] -= allocation[i][j]
        return allocation

    def algorithme_stepping_stone(self, allocation, couts_transport):
        # Implement the "Stepping Stone" algorithm here
        # This is a placeholder for the actual implementation
        return allocation  # Return the same allocation for now

    def moindre_cout_callback(self):
        num_magazines = simpledialog.askinteger("Input", "Nombre de magasins :")
        num_usines = simpledialog.askinteger("Input", "Nombre d'usines :")
        
        couts_transport, capacites_usines, demandes_magazines = self.generate_random_data(num_magazines, num_usines)
        
        allocation = self.algorithme_moindre_cout(capacites_usines.copy(), demandes_magazines.copy(), couts_transport)
        cout_total = self.calculer_cout_total(allocation, couts_transport)
        
        result_text = f"Coût total : {cout_total}\nAllocation :\n{allocation}"
        self.display_results(result_text)

    def nord_ouest_callback(self):
        num_magazines = simpledialog.askinteger("Input", "Nombre de magasins :")
        num_usines = simpledialog.askinteger("Input", "Nombre d'usines :")
        
        couts_transport, capacites_usines, demandes_magazines = self.generate_random_data(num_magazines, num_usines)
        
        allocation = self.algorithme_nord_ouest(capacites_usines.copy(), demandes_magazines.copy(), couts_transport)
        cout_total = self.calculer_cout_total(allocation, couts_transport)
        
        result_text = f"Coût total : {cout_total}\nAllocation :\n{allocation}"
        self.display_results(result_text)

    def stepping_stone_callback(self):
        num_magazines = simpledialog.askinteger("Input", "Nombre de magasins :")
        num_usines = simpledialog.askinteger("Input", "Nombre d'usines :")
        
        couts_transport, capacites_usines, demandes_magazines = self.generate_random_data(num_magazines, num_usines)
        
        initial_allocation = self.algorithme_nord_ouest(capacites_usines.copy(), demandes_magazines.copy(), couts_transport)
        optimized_allocation = self.algorithme_stepping_stone(initial_allocation, couts_transport)
        cout_total = self.calculer_cout_total(optimized_allocation, couts_transport)
        
        result_text = f"Coût total après optimisation : {cout_total}\nAllocation optimisée :\n{optimized_allocation}"
        self.display_results(result_text)

    def calculer_cout_total(self, allocation, couts_transport):
        # Calculate the total transportation cost based on the allocation and costs
        total_cost = 0
        for i in range(len(allocation)):
            for j in range(len(allocation[i])):
                total_cost += allocation[i][j] * couts_transport[i][j]
        return total_cost

    def display_results(self, result_text):
        # Display results in the Tkinter GUI
        self.code_display.delete(1.0, tk.END)  # Clear previous results
        self.code_display.insert(tk.END, result_text)  # Insert new results

    def bellman_ford_callback(self):
        try:
            n = simpledialog.askinteger("Input", "Entrez le nombre de sommets pour le graphe :", parent=self.root, minvalue=2)
            if n is None:  # User cancelled
                return
            
            G = self.generer_graphe_aleatoire(n)
            start_node = simpledialog.askstring("Input", "Entrez le sommet de départ (ex: A, B, C, ...):", parent=self.root)
            
            if start_node not in G.nodes:
                messagebox.showerror("Error", "Le sommet de départ n'est pas valide.")
                return
            
            distances, predecessors = self.bellman_ford(G, start_node)

            result_text = f"Distances depuis le sommet {start_node}:\n"
            for node, distance in distances.items():
                result_text += f"{node}: {distance}\n"

            self.display_results(result_text)
            self.visualize_bellman_ford(G, distances, start_node)

        except Exception as e:
            messagebox.showerror("Error", f"Une erreur s'est produite: {str(e)}")

    def bellman_ford(self, G, start_node):
        # Initialize distances and predecessors
        distances = {node: float('inf') for node in G.nodes()}
        distances[start_node] = 0
        predecessors = {node: None for node in G.nodes()}

        # Relax edges up to |V| - 1 times
        for _ in range(len(G.nodes()) - 1):
            for u, v, data in G.edges(data=True):
                weight = data['weight']
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u

        # Check for negative-weight cycles
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            if distances[u] + weight < distances[v]:
                raise ValueError("Le graphe contient un cycle de poids négatif.")

        return distances, predecessors

    def visualize_bellman_ford(self, G, distances, start_node):
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'weight')

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Highlight the shortest paths
        for node, distance in distances.items():
            if distance < float('inf'):
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='green' if node != start_node else 'yellow')

        plt.title("Visualisation de l'algorithme de Bellman-Ford")
        plt.show()

    def ford_fulkerson_callback(self):
        try:
            num_vertices = simpledialog.askinteger("Input", "Entrez le nombre de sommets :", parent=self.root, minvalue=2)
            if num_vertices is None:  # User cancelled
                return
            
            g = Graph(num_vertices)

            # Generate random edges with capacities
            for _ in range(num_vertices * 2):  # Randomly create edges
                u = random.randint(0, num_vertices - 1)
                v = random.randint(0, num_vertices - 1)
                if u != v:  # Avoid self-loops
                    capacity = random.randint(1, 20)  # Random capacity between 1 and 20
                    g.add_edge(u, v, capacity)
                    print(f"Edge added: {u} -> {v} with capacity {capacity}")  # Debug line

            source = simpledialog.askinteger("Input", "Entrez le sommet source :", parent=self.root, minvalue=0, maxvalue=num_vertices - 1)
            sink = simpledialog.askinteger("Input", "Entrez le sommet puits :", parent=self.root, minvalue=0, maxvalue=num_vertices - 1)

            if source == sink:
                messagebox.showerror("Error", "Le sommet source et le sommet puits ne peuvent pas être identiques.")
                return

            max_flow = g.ford_fulkerson(source, sink)
            min_cut_edges = g.min_cut(source)

            result_text = f"Le flot maximum est : {max_flow}\nCoupe minimale : {min_cut_edges}"
            self.display_results(result_text)

        except Exception as e:
            messagebox.showerror("Error", f"Une erreur s'est produite: {str(e)}")

def main():
    root = tk.Tk()
    app = OperationsResearchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()