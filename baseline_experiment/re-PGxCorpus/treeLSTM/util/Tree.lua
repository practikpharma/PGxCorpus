--[[

  A basic tree structure.

--]]

local Tree = torch.class('treelstm.Tree')

function Tree:__init()
  self.parent = nil
  self.num_children = 0
  self.children = {}
end

function Tree:add_child(c)
  c.parent = self
  self.num_children = self.num_children + 1
  self.children[self.num_children] = c
end

function Tree:size()
  if self._size ~= nil then return self._size end
  local size = 1
  for i = 1, self.num_children do
    size = size + self.children[i]:size()
  end
  self._size = size
  return size
end

function Tree:depth()
  local depth = 0
  if self.num_children > 0 then
    for i = 1, self.num_children do
      local child_depth = self.children[i]:depth()
      if child_depth > depth then
        depth = child_depth
      end
    end
    depth = depth + 1
  end
  return depth
end

local function depth_first_preorder(tree, nodes)
  if tree == nil then
    return
  end
  table.insert(nodes, tree)
  for i = 1, tree.num_children do
    depth_first_preorder(tree.children[i], nodes)
  end
end

function Tree:depth_first_preorder()
  local nodes = {}
  depth_first_preorder(self, nodes)
  return nodes
end

--From Joel
local function print_tree(tree, st)
   print(st .. tree.idx)-- .. " " .. (tree.dep and tree.dep .. " " or " ") .. (tree.word and tree.word .. " " or " ") .. (tree.pos and tree.pos .. " " or " ") .. (tree.ent and " (" .. tree.ent .. ")" or "") .. " parent: " .. (tree.parent and tree.parent.idx .. " " or "ROOT ") .. (tree.depth and tree.depth or ""))
   for i=1, tree.num_children do
      print_tree(tree.children[i], st .. "\t")
   end
end

-- --From Joel
function Tree:print()
   print_tree(self, "")
end

function Tree:clean()
   self.visited = false
   for i = 1, self.num_children do
      self.children[i]:clean()
   end
end

function Tree:dist(from, to)
   --print("visinting node " .. self.idx)
   self.visited = true
   if self.ent == to then return 0 end

   for i = 1, self.num_children do
      if not self.children[i].visited then
	 local dist = self.children[i]:dist(from, to)
	 if dist ~= -1 then return dist + 1 end
      end
   end

   if self.parent and not self.parent.visited then return 1 + self.parent:dist(from, to) end

   return -1 --ent not found, sons and parent already visited
end


function Tree:_isHead(from, to)
   if self.ent==to then
      return true
   else
      local res = false
      for i = 1, self.num_children do
	 res = res or self.children[i]:_isHead(from, to)
      end
      return res
   end
end

function Tree:_isHeaded(from, to)
   if self.ent==to then
      return true
   else
      return self.parent:_isHeaded(from, to)
   end
end


function Tree:isHead(from, to)
   return self:_isHead(from, to) or self:_isHead(from, to)
end

function Tree:shortest_path(from, to)
   --print("visinting node " .. self.idx)
   self.visited = true
   if self.ent == to then
      --print("found")
      return {self}
   end

   for i = 1, self.num_children do
      if not self.children[i].visited then
	 local path = self.children[i]:shortest_path(from, to)
	 if path then
	    table.insert(path, self)
	    --print("youpi " .. #path)
	    return path
	 end
      end
   end

   if self.parent and not self.parent.visited then
      local path = self.parent:shortest_path(from, to)
      if path then
	 table.insert(path, self)
	 --print("youpi " .. #path)
	 return path
      end
   end
end

function Tree:set_depth(depth)
   self.depth = depth
   for i = 1, self.num_children do
      self.children[i]:set_depth(depth+1)
   end
end

function Tree:group_pos(tab_group_pos)
   self.grouppos = tab_group_pos[self.pos]
   for i = 1, self.num_children do
      self.children[i]:group_pos(tab_group_pos)
   end
end

function Tree:group_dep(tab_group_dep)
   self.groupdep = tab_group_dep[self.dep]
   for i = 1, self.num_children do
      self.children[i]:group_dep(tab_group_dep)
   end
end


function Tree:path_to_fingerprint(path)
   -- print(#path)
   -- for i=1,#path do
   --    print(path[i].idx, path[i].word, path[i].dep)
   -- end
   
   local min, highest = math.huge
   local minind
   for i=1,#path do
      if path[i].depth<min then min=path[i].depth; highest=path[i];minind=i end
   end

   local fingerprint, tnirpregnif = "", ""
   
   if true then
      for i=1,#path do
	 if path[i]==highest then
	    fingerprint = fingerprint .. " " .. path[i].pos
	    tnirpregnif = path[i].pos .. " " .. tnirpregnif
	 elseif (path[i+1] and path[i+1].depth<path[i].depth) or path[i].depth<path[i-1].depth then
	    fingerprint = (fingerprint=="" and '' or fingerprint .. " ") .. path[i].pos .. " " .. path[i].dep
	    tnirpregnif = path[i].dep .. " " .. path[i].pos .. " " .. tnirpregnif
	 else
	    fingerprint = fingerprint .. " " .. path[i].dep .. " " .. path[i].pos
	    tnirpregnif = path[i].pos .. " " .. path[i].dep .. " " .. tnirpregnif
	 end
      end
      fingerprint = fingerprint:gsub(" +$", "")
      tnirpregnif = tnirpregnif:gsub(" +$", "")
      fingerprint = fingerprint:gsub("^ +", "")
      tnirpregnif = tnirpregnif:gsub("^ +", "")
   end

   --print(fingerprint,tnirpregnif)
   
   
   if true then
      local right,left = "", ""
      if minind~=1 then
	 for i=minind-1, 1, -1 do
	    --print("-1 " .. i)
	    left = left .. " " .. path[i].dep .. " " .. path[i].pos .. " " --.. path[i].word .. " "
	 end
      end
      if minind~=#path then
	 for i=minind+1,#path do
	    --print("+1 " .. i)
	    right = right .. " " .. path[i].dep .. " " .. path[i].pos .. " "-- .. path[i].word .. " "
	 end
      end
      -- print("===")
      -- print(minind)
      -- print(path[minind].dep)
      fingerprint = path[minind].pos .. " +" .. left .. " +" .. right --" " .. path[minind].word ..
      tnirpregnif = path[minind].pos .. " " .. " +" .. right .. " +" .. left --path[minind].word ..
      fingerprint = fingerprint:gsub(" +", " ")
      tnirpregnif = tnirpregnif:gsub(" +", " ")
      fingerprint = fingerprint:gsub(" $", "")
      tnirpregnif = tnirpregnif:gsub(" $", "")
      fingerprint = fingerprint:gsub("^ +", "")
      tnirpregnif = tnirpregnif:gsub("^ +", "")
   end
   --exit()
   --io.read()
   return fingerprint, tnirpregnif
   
end


function Tree:path_to_fingerprint_words(path)
   local min, highest = math.huge
   local minind
   for i=1,#path do
      if path[i].depth<min then min=path[i].depth; highest=path[i];minind=i end
   end

   local fingerprint, tnirpregnif = "", ""
   
   for i=1,#path do
      if path[i]==highest then
	 fingerprint = fingerprint .. " *" .. path[i].word .. "*"
	 tnirpregnif = "*" .. path[i].word .. "* " .. tnirpregnif
      elseif (path[i+1] and path[i+1].depth<path[i].depth) or path[i].depth<path[i-1].depth then
	 fingerprint = (fingerprint=="" and '' or fingerprint .. " ") .. path[i].word .. " " .. path[i].dep
	 tnirpregnif = path[i].dep .. " " .. path[i].word .. " " .. tnirpregnif
      else
	 fingerprint = fingerprint .. " " .. path[i].dep .. " " .. path[i].word
	 tnirpregnif = path[i].word .. " " .. path[i].dep .. " " .. tnirpregnif
      end
   end

   fingerprint = fingerprint:gsub(" +$", "")
   tnirpregnif = tnirpregnif:gsub(" +$", "")
   fingerprint = fingerprint:gsub("^ +", "")
   tnirpregnif = tnirpregnif:gsub("^ +", "")
   

   if true then
      local right,left = "", ""
      if minind~=1 then
	 for i=minind-1, 1, -1 do
	    --print("-1 " .. i)
	    left = left .. " " .. path[i].dep .. " " .. path[i].word .. " " --.. path[i].word .. " "
	 end
      end
      if minind~=#path then
	 for i=minind+1,#path do
	    --print("+1 " .. i)
	    right = right .. " " .. path[i].dep .. " " .. path[i].word .. " "-- .. path[i].word .. " "
	 end
      end
      -- print("===")
      -- print(minind)
      -- print(path[minind].dep)
      fingerprint = path[minind].word .. " +" .. left .. " +" .. right --" " .. path[minind].word ..
      tnirpregnif = path[minind].word .. " " .. " +" .. right .. " +" .. left --path[minind].word ..
      fingerprint = fingerprint:gsub(" +", " ")
      tnirpregnif = tnirpregnif:gsub(" +", " ")
      fingerprint = fingerprint:gsub(" $", "")
      tnirpregnif = tnirpregnif:gsub(" $", "")
      fingerprint = fingerprint:gsub("^ +", "")
      tnirpregnif = tnirpregnif:gsub("^ +", "")
      
   end

   
   
   return fingerprint, tnirpregnif
   
end


function isInPath(path, node)
   local res = false
   for i=1,#path do
      if node==path[i] then
	 res = true
      end
   end
   return res
end

function Tree:path_to_fingerprint_level1(path)
   local min, highest = math.huge
   local minind
   for i=1,#path do
      if path[i].depth<min then min=path[i].depth; highest=path[i];minind=i end
   end

   local fingerprint, tnirpregnif = "", ""
   
   local right,left = "", ""
   if minind~=1 then
      for i=minind-1, 1, -1 do
	 --print("-1 " .. i)
	 left = left .. " " .. path[i].dep .. " " .. path[i].pos .. " [ " --.. path[i].word .. " "
	 for j=1,#path[i].children do
	    --print(j .. " " .. path[i].children[j].idx .. " " .. (isInPath(path, path[i].children[j]) and "true" or "false"))
	    --if not isInPath(path, path[i].children[j]) then
	    left = left .. " " .. path[i].children[j].dep .. " " .. path[i].children[j].pos .. " "
	    --end
	 end
	 left = left .. " ] "
      end
   end
   if minind~=#path then
      for i=minind+1,#path do
	 --print("+1 " .. i)
	 right = right .. " " .. path[i].dep .. " " .. path[i].pos .. " [ "-- .. path[i].word .. " "
	 for j=1,#path[i].children do
	    --print(j .. " " .. path[i].children[j].idx .. " " .. (isInPath(path, path[i].children[j]) and "true" or "false"))
	    --if not isInPath(path, path[i].children[j]) then
	    right = right .. " " .. path[i].children[j].dep .. " " .. path[i].children[j].pos .. " "
	    --end
	 end
	 right = right .. " ] "

      end
   end

   local lca = path[minind].pos .. " [ "
   for j=1,#path[minind].children do
      lca = lca .. " " .. path[minind].children[j].dep .. " " .. path[minind].children[j].pos .. " "
   end
   lca = lca .. " ] "
   
   -- print("===")
   -- print(minind)
   -- print(path[minind].dep)
   fingerprint = lca .. " +" .. left .. " +" .. right --" " .. path[minind].word ..
   tnirpregnif = lca .. " " .. " +" .. right .. " +" .. left --path[minind].word ..
   fingerprint = fingerprint:gsub(" +", " ")
   tnirpregnif = tnirpregnif:gsub(" +", " ")
   fingerprint = fingerprint:gsub(" $", "")
   tnirpregnif = tnirpregnif:gsub(" $", "")
   fingerprint = fingerprint:gsub("^ +", "")
   tnirpregnif = tnirpregnif:gsub("^ +", "")
   
   
   return fingerprint, tnirpregnif
   
end


function Tree:path_to_3fingerprint(path)
   local min, highest = math.huge
   local minind
   for i=1,#path do
      if path[i].depth<min then min=path[i].depth; highest=path[i];minind=i end
   end

   local fingerprint, tnirpregnif = "", ""

   if true then
      local right,left = "", ""
      if minind~=1 then
	 left = left .. " " .. path[minind-1].dep --" " .. path[i].pos
      end
      if minind~=#path then
	 --print("+1 " .. i)
	 right = right .. " " .. path[minind+1].dep --" " .. path[i].pos
      end

      fingerprint = path[minind].pos .. " +" .. left .. " +" .. right --" " .. path[minind].word ..
      tnirpregnif = path[minind].pos .. " " .. " +" .. right .. " +" .. left --path[minind].word ..
      fingerprint = fingerprint:gsub(" +", " ")
      tnirpregnif = tnirpregnif:gsub(" +", " ")
      fingerprint = fingerprint:gsub(" $", "")
      tnirpregnif = tnirpregnif:gsub(" $", "")
      fingerprint = fingerprint:gsub("^ +", "")
      tnirpregnif = tnirpregnif:gsub("^ +", "")
   end
   --exit()
   --io.read()
   return fingerprint, tnirpregnif
   
end



function Tree:path_to_fingerprint_group(path, group_pos, group_dep)
   local min, highest = math.huge
   local minind
   for i=1,#path do
      if path[i].depth<min then min=path[i].depth; highest=path[i];minind=i end
   end

   local fingerprint, tnirpregnif = "", ""

   local right,left = "", ""
   if minind~=1 then
      for i=minind-1, 1, -1 do
	 -- print("-1 " .. i)
	 -- print(path[i].pos)
	 -- print(path[i].grouppos)
	 -- print(group_pos and path[i].grouppos or path[i].pos)
	 local new_dep

	 if group_dep and path[i].groupdep then
	    new_dep = path[i].groupdep .. "_1"
	 else
	    new_dep = path[i].dep .. "_1"
	 end
	 local new_pos
	 if group_pos and path[i].grouppos then
	    new_pos = path[i].grouppos .. "_1"
	 else
	    new_pos = path[i].pos .. "_1"
	 end
	 left = left .. " " .. new_dep .. " " .. new_pos .. " " --.. path[i].word .. " "
      end
   end
   if minind~=#path then
      for i=minind+1,#path do
	 --print("+1 " .. i)
	 -- print(path[i].pos)
	 -- print(path[i].grouppos)
	 -- print(group_pos and path[i].grouppos or path[i].pos)
	 -- print("===")
	 local new_dep
	 if group_dep and path[i].groupdep then
	    new_dep = path[i].groupdep .. "_1"
	 else
	    new_dep = path[i].dep .. "_1"
	 end
	 local new_pos
	 if group_pos and path[i].grouppos then
	    new_pos = path[i].grouppos .. "_1"
	 else
	    new_pos = path[i].pos .. "_1"
	 end
	 right = right .. " " .. new_dep .. " " .. new_pos .. " "-- .. path[i].word .. " "
      end
   end
   -- print("===")
   -- print(minind)
   -- print(path[minind].dep)
   local new_pos
   if group_pos and path[minind].grouppos then
      new_pos = path[minind].grouppos .. "_1"
   else
      new_pos = path[minind].pos .. "_1"
   end
	 
   fingerprint = new_pos .. " +" .. left .. " +" .. right --" " .. path[minind].word ..
   tnirpregnif = new_pos .. " " .. " +" .. right .. " +" .. left --path[minind].word ..
   fingerprint = fingerprint:gsub(" +", " ")
   tnirpregnif = tnirpregnif:gsub(" +", " ")
   fingerprint = fingerprint:gsub(" $", "")
   tnirpregnif = tnirpregnif:gsub(" $", "")
   fingerprint = fingerprint:gsub("^ +", "")
   tnirpregnif = tnirpregnif:gsub("^ +", "")

   --exit()
   --io.read()
   return fingerprint, tnirpregnif
   
end


--only pos tags
function Tree:path_to_fingerprint_posordep_group(path, group_pos, group_dep, takepos, takedep)
   local min, highest = math.huge
   local minind
   for i=1,#path do
      if path[i].depth<min then min=path[i].depth; highest=path[i];minind=i end
   end

   local fingerprint, tnirpregnif = "", ""

   local right,left = "", ""
   if minind~=1 then
      for i=minind-1, 1, -1 do
	 local new_dep
	 if group_dep and path[i].groupdep then
	    new_dep = path[i].groupdep .. "_1"
	 else
	    new_dep = path[i].dep .. "_1"
	 end
	 local new_pos
	 if group_pos and path[i].grouppos then
	    new_pos = path[i].grouppos .. "_1"
	 else
	    new_pos = path[i].pos .. "_1"
	 end
	 left = left .. " " .. (takedep and (new_dep .. " ") or "") .. (takepos and (new_pos .. " ") or "") --.. path[i].word .. " "
      end
   end
   if minind~=#path then
      for i=minind+1,#path do
	 --print("+1 " .. i)
	 -- print(path[i].pos)
	 -- print(path[i].grouppos)
	 -- print(group_pos and path[i].grouppos or path[i].pos)
	 -- print("===")
	 local new_dep
	 if group_dep and path[i].groupdep then
	    new_dep = path[i].groupdep .. "_1"
	 else
	    new_dep = path[i].dep .. "_1"
	 end
	 local new_pos
	 if group_pos and path[i].grouppos then
	    new_pos = path[i].grouppos .. "_1"
	 else
	    new_pos = path[i].pos .. "_1"
	 end
	 right = right .. " " .. (takedep and (new_dep .. " ") or "") .. (takepos and (new_pos .. " ") or "") -- .. path[i].word .. " "
      end
   end
   -- print("===")
   -- print(minind)
   -- print(path[minind].dep)
   local new_pos
   if group_pos and path[minind].grouppos then
      new_pos = path[minind].grouppos .. "_1"
   else
      new_pos = path[minind].pos .. "_1"
   end
	 
   fingerprint = (takepos and (new_pos .. " ") or "") .. " +" .. left .. " +" .. right --" " .. path[minind].word ..
   tnirpregnif = (takepos and (new_pos .. " ") or "") .. " +" .. right .. " +" .. left --path[minind].word ..
   fingerprint = fingerprint:gsub(" +", " ")
   tnirpregnif = tnirpregnif:gsub(" +", " ")
   fingerprint = fingerprint:gsub(" $", "")
   tnirpregnif = tnirpregnif:gsub(" $", "")
   fingerprint = fingerprint:gsub("^ +", "")
   tnirpregnif = tnirpregnif:gsub("^ +", "")

   --exit()
   --io.read()
   return fingerprint, tnirpregnif
   
end
