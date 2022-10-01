# TheWelfareProject
A linear solver for maximising social welfare, given a valuation matrix of dimension = (number of agents, number of items). The library uses python as the language of choice and the framework for optimisation is Google OR-TOOLS (SCIP).
Given a valuation profile say $$V = \{v_{ij}\}$$ we first need to find a bipartite matching of the agents to the items and then compute the welfare using a simple addition over all the allocated bundles. 

The constraints used right now include: 
1. budget constraint (1)
2. envy constraint for all pairs of agents ($$n^2 - n$$)
3. assignment constraint for all the items ($$m$$) 
4. optional constraint of each subsidy to be in range 0 to 1. 


The tool can be used to get the final allocation, objective, subsidies. 
