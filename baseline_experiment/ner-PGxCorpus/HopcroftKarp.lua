require("torch")

local NIL = 0
local INF = math.huge

local graph = {}

function graph:graph(m,n)
   self.m = m
   self.n = n
   self.adj = {}
   for i=1,m+1 do
      self.adj[i] = {}
   end
   return self
end

function graph:addEdge(u,v)
   table.insert(self.adj[u],v)
   --table.insert(self.adj[v],u)
end


function graph:bfs()

   local Q = {}

   -- First layer of vertices (set distance as 0)
   --print("======================= start bfs =====================")
   for u=1,self.m do
      --io.write(u .. " ")
      if self.pairU[u]==NIL then
	 --print("toto");
	 self.dist[u]=0
	 table.insert(Q, u)
      else
	 self.dist[u]=INF
      end
   end
   
   self.dist[NIL] = INF
   
   while #Q~=0 do
      --print("while")
      local u = table.remove(Q,1)
      --print("node " .. u)
      
      -- If this node is not NIL and can provide a shorter path to NIL
      
      if self.dist[u]<self.dist[NIL] then

	 for i=1,#self.adj[u] do
	    local v = self.adj[u][i]
	    --print("v " .. v)
	    --print(self.pairV)
	    --print(self.pairV[v] .. " " .. self.dist[self.pairV[v]] .. " INF")
	    if (self.dist[self.pairV[v]] == INF) then
	       --print("INF")
	       self.dist[self.pairV[v]] = self.dist[u] + 1;
	       table.insert(Q,self.pairV[v])
	    end
	 end
	 
      end
      
   end
   
   --print("return " .. ((self.dist[NIL]~=INF) and "true" or "false") .."  (" .. self.dist[NIL] .. ")")
   return (self.dist[NIL] ~= INF)
   
end


function graph:dfs(u)
   --print("====================start DFS===================")
   if u~=NIL then--check that
      for i=1,#self.adj[u] do
	 local v = self.adj[u][i]
	 --print("v " .. v)
	 
	 if self.dist[self.pairV[v]] == self.dist[u]+1 then
	    if graph:dfs(self.pairV[v]) then
	       self.pairV[v]=u
	       self.pairU[u]=v
	       return true
	    end
	 end
      end
      
      -- If there is no augmenting path beginning with u.
      self.dist[u] = INF;
      return false;

   end
   return true
end


function graph:hopcroftKarp()
   --pairU[u] stores pair of u in matching where u
   --is a vertex on left side of Bipartite Graph.
   --If u doesn't have any pair, then pairU[u] is NIL
   self.pairU = {}; for i=0,self.m do self.pairU[i]=0 end--torch.Tensor(self.m+1):fill(0)

   --pairV[v] stores pair of v in matching. If v
   --doesn't have any pair, then pairU[v] is NIL
   self.pairV = {}; for i=0, self.n do self.pairV[i]=0 end--torch.Tensor(self.n+1):fill(0)

   --dist[u] stores distance of left side vertices
   --dist[u] is one more than dist[u'] if u is next
   --to u'in augmenting path
   self.dist = {}; for i=0, self.m do self.dist[i]=0 end-- torch.Tensor(self.m+1):fill(0);
   

   --Initialize  pair of all vertices
   for u=0, self.m do
      self.pairU[u] = NIL
   end
   for v=0, self.n do
      self.pairV[v] = NIL
   end

   local result = 0

   while (self:bfs()) do
      --print("bfs is true")
      for u=1,self.m do
	 if self.pairU[u]==NIL and self:dfs(u) then
	    result = result + 1
	 end
      end
   end
   return result
end

-- local g 

-- if false then
--    g = graph:graph(4,4)
--    g:addEdge(1,2)
--    g:addEdge(1, 3);
--    g:addEdge(2, 1);
--    g:addEdge(3, 2);
--    g:addEdge(4, 2);
--    g:addEdge(4, 4);
-- else
--    if false then
--       g = graph:graph(4,4)
--       g:addEdge(1,1);
--       g:addEdge(2,1);
--       g:addEdge(3,2);
--       g:addEdge(3,3);
--       g:addEdge(3,4);
--       g:addEdge(4,4);
--    else
--       if false then
-- 	 g = graph:graph(3,1)
-- 	 g:addEdge(1,1);
-- 	 g:addEdge(2,1);
-- 	 g:addEdge(3,1);
--       else
-- 	 g = graph:graph(1,3)
-- 	 g:addEdge(1,1);
-- 	 g:addEdge(1,2);
-- 	 g:addEdge(1,3);
--       end
--    end
-- end

-- local n = g:hopcroftKarp()

-- print("Size of maximum matching is " .. n)
return graph
