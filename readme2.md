There are n variates, m nodes. The way that things are set up right now, m = k x n, and each node corresponds to exactly one variable. For each variable there are multiple nodes. Each node has a particular physical locations (I think) which are learned and the graph is between these nodes. So, this means that as output we have causal connectivity between nodes that:
1. Correspond to different variates
2. Have different spatial locations

See figure 15 in the paper. This is the kind of output that we get.

The code gets slow when we have a lot of variates and a lot of nodes. So for now, I just kept 4 variates: PM10, PM2.5, COPD, and TB. And 2 nodes per variate (that is 8 nodes). The output is crude. See map.png

G_pred.pt is the graph, center.pt gives the locations of each node.
scale.pt also gives the shape of these nodes. alpha.pt maps nodes to variates. F_pred.pt tells the influence each spatial grid has on each of the center, so we can also color the locations by the values present in this to understand what has affected our causal calculations.

Weirdness:
1. the code saves centers as a tensor of shape (num_variates, num_nodes, 1, 2). Why is it like this? It should just be (num_nodes, 2). And the rest could be figured out using alpha.pt.
2. F_pred.pt similarly is (num_variates, num_nodes, nx, ny), but most of the entries are zeros (since they are invalid indices, for example if k = 2, then variate with variate index n=0 corresponds only to nodes with indices m={0,1}, then entry(0,2,:,:) are zeros/invalid, similarly n=i corresponds to m={2*i,2*i+1} and the other entries are zeros).
3. Even weirder, for centers, if for any value of n, I denote valid entries as m and invalid entries as im, then entries (n,m,1,2) are non-zero as expected, but entries (n,im,1,2) are also non-zeroes! I have filtered them out as of now, hopefully it is correct. Look at the code in plot_map.py for details of my interpretation.
