require 'torch'

function _setlevel(node, contiguous)
   --print("---- " .. node[3])
   if #node.sons==0 then
      if contiguous and #node[1]>1 then --discontiguous entity
	 --print(node[3] .. ' 1')
	 node.level = 0
	 return node.level
      else
	 --print(node[3] .. ' 2')
	 node.level=1
	 return node.level
      end
   else
      local max_son = 0
      for i=1,#node.sons do
	 max_son = math.max(max_son, _setlevel(node.sons[i], contiguous))
      end
      if contiguous and #node[1]>1 then --discontiguous entity
	 --print(node[3] .. ' 3')
	 node.level = 0
	 return max_son
      else
	 --print(node[3] .. ' 4')
	 node.level = 1 + max_son
	 return 1 + max_son
      end
   end
end
   
--compute level in dag
--contiguous option allow to discard discontiguous entities
function setlevel(ent, contiguous)
   --printdag(ent)
   --for i=1,#ent do
   --   print(ent[1])
   --end
   for i=1,#ent do
      --print("=============" .. i)
      _setlevel(ent[i], contiguous)
   end
end


function _load_entity_indices(ents, starts, ends)
   --print("============")
   --print(ents)
   
   --printw(words[i], wordhash)
   --print(words.sent[i])
      --print(starts)
   --print(ends)
   
   for j=1,#ents do
      --idx: word indices corresponding to the entity
      local idx = {}
      --print(ents[j])
      for e=1,#ents[j][1] do
	 local _start, _end
	 for _s=1, starts:size(1) do
	    --print(starts[_s]+(_s-1) .. " " .. ents[j][1][e][1])
	    -- (+ _s-1) to include spaces between words
	    if starts[_s]+(_s-1)==ents[j][1][e][1] then --start found
	       --print("start found " .. _s)
	       _start = _s
	    end
	 end
	 assert(_start, "_start not found")
	 --print("start " .. _start)
	 for _e=_start, ends:size(1) do
	    --print(ends[_e]+_e .. " " .. ents[j][1][e][2])
	    -- (+ _e) to include spaces between words
	    if ends[_e]+_e==ents[j][1][e][2] then
	       --print("end found " .. _e)
	       _end = _e
	    end
	 end
	 assert(_end, "_end not found")
	 --print(_start .. " " .. _end)
	 for i=_start, _end do
	    table.insert(idx, i)
	 end
      end
      
      ents[j][5] = torch.Tensor(idx)
      --print(ents[j])
      --io.read()
   end
end

local function load_entity_indices(entities, words, starts, ends, wordhash)
   assert(#entities==#words.idx and #entities==#starts and #entities==#ends, #entities .. " " .. #words.idx .. " " .. #starts .. " " .. #ends)

   for i=1,#entities do
      _load_entity_indices(entities[i], starts[i], ends[i])
   end
   
end


local function loadstartend(pathdata, feature, maxload)
   print("loading startend in " .. pathdata)
   local starts = {}
   local ends = {}
   local _break = false

   local handle = io.popen("find " .. pathdata .. " -name '*.txt' | sort")
   local filename = handle:read()
   while filename do
      --print(string.format('loading startend for <%s>', filename))
      for line in io.lines(filename) do
	 --print(line)
	 if maxload and maxload > 0 and maxload == #starts then
	    print("break")
	    _break = true
	    break
	 end
	 if line~="" then
	    local s, e = {}, {}
	    local i = 0
	    for word in line:gmatch('(%S+)') do
	       table.insert(s, i)
	       --print(word)
	       --print(#word)
	       i = i + #word
	       table.insert(e, i-1)
	    end
	    table.insert(starts, torch.IntTensor(s))
	    table.insert(ends, torch.IntTensor(e))
	    --local t = torch.IntTensor(s) 
	    --print(t:resize(1,t:size(1)))
	    --local t2 = torch.IntTensor(e) 
	    --print(t2:resize(1,t2:size(1)))
	 end
	 --io.read()
      end
      filename = handle:read()
      if _break then print("break 1"); break end
   end
   handle:close()
   return starts, ends
end


function loaddag(entities)
   local indices = {}
   for i=1,#entities do
      getdag(entities[i])
   end
   return indices
end

--build an inclusion dag (directed acyclic graph)
function getdag(ent)
   for ent1=1,#ent do
      ent[ent1].sons = {}
   end
   
   for ent1=1,#ent do
      for ent2=1,#ent do
	 --print("=====================================> " .. ent1 .. " " .. ent2)
	 if ent1~=ent2 and is_included(ent[ent1][1], ent[ent2][1]) then
	    --print(ent1 .. " is included in " .. ent2)
	    table.insert(ent[ent2].sons, ent[ent1])
	    --io.read()
	 end
	 
      end
   end
end

--return true if ent1 in included in ent2
function is_included(ent1, ent2)
   local res = true
   -- print("====")
   -- print(ent1)
   -- print(ent2)
   
   for i=1, #ent1 do
      local found  = false
      for j=1,#ent2 do
	 local b1, e1 = ent1[i][1], ent1[i][2] --note: change that to include uncontiguous entities
	 local b2, e2 = ent2[j][1], ent2[j][2]
	 if type(b1)=="string" then error("") end
	 if b2<=b1 and e2>=e1 then found = true end
      end
      res = res and found
   end
   -- print("is_included")
   -- print(res)
   return res
end



local function tree2tree(trees)
   --print(string.format('tree2tree'))
   local newtrees = {}
   for i=1,#trees do
      local tree = trees[i]
      --print(tree)
      local reps = {}
      local j = 1
      while j<#tree do
	 local size = tree[j]
	 j = j+1
	 local head = treelstm.Tree()
	 head.idx = tree[j]
	 j = j + 1
	 for k=1,size-1 do
	    if tree[j]<1000 then
	       local son = treelstm.Tree()
	       son.idx = tree[j]
	       head:add_child(son)
	    else
	       head:add_child(reps[ tree[j]-1000 ])
	    end
	    j = j + 1
	 end
	 table.insert(reps, head)
      end
      for i=1,#reps do
	 --print(i)
	 --reps[i]:print()
      end
      -- print("==============")
      -- reps[#reps]:print()
      table.insert(newtrees, reps[#reps])
      --io.read()
   end
   return newtrees
end

local function loadtrees(filename, maxid)
   print(string.format('loading <%s>', filename))
   local trees = {}
   for line in io.lines(filename) do
      if line~="" then
	 if maxload and maxload > 0 and maxload == #trees then
	    print("breakdata10")
	    break
	 end
	 local tab = {}
	 for word in line:gmatch('(%d+)') do
	    table.insert(tab, tonumber(word))
	 end
	 table.insert(trees, tab)
      end
   end
   return trees
end

local function _loadhash(filename, maxidx)
   print(string.format('loading <%s>', filename))
   local hash = {}
   local idx = 0
   for key in io.lines(filename) do
      idx = idx + 1
      if maxidx and maxidx > 0 and idx > maxidx then
         break
      end
      table.insert(hash, key)
      hash[key] = idx
   end
   return hash
end

local function _addhash(filename, hash)
   print(string.format('adding <%s> to hash', filename))
   local idx = #hash
   local _added, _present = 0, 0
   for key in io.lines(filename) do
      if not hash[key] then
	 _added = _added + 1
	 idx = idx + 1
	 table.insert(hash, key)
	 hash[key] = idx
      else
	 _present = _present + 1
      end
   end
   print(_added .. " words added, " .. _present .. " words already in hash")
   return hash
end

local function wordfeature(word)
   word = word:lower()
   word = word:gsub('%d+', '0')
   return word
end

local function loadindices(filename, maxload)
   print(string.format('loading <%s>', filename))
   local res = {}
   for line in io.lines(filename) do
      table.insert(res, line)
   end
   return res
end

local function loadwords(pathdata, hash, addraw, feature, maxload)
   maxidx = maxidx or #hash
   local lines = addraw and {} or nil
   local indices = {}
   local sentences = {}
   local _break = false
   
   local handle = io.popen("find " .. pathdata .. " -name '*.txt' | sort")
   local filename = handle:read()
   while filename do
      --print(string.format('loading <%s>', filename))
      for line in io.lines(filename) do
	 --print(line)
	 table.insert(sentences, line)
	 if line~="" then
	    if maxload and maxload > 0 and maxload == #indices then
	       print("breakdata2")
	       _break = true
	       break
	    end
	    local words = {}
	    local wordsidx = {}
	    for word in line:gmatch('(%S+)') do
	       if addraw then
		  table.insert(words, word)
	       end
	       table.insert(wordsidx, hash[feature and feature(word) or word] or hash.UNK)
	    end
	    if addraw then
	       table.insert(lines, words)
	    end   
	    
	    table.insert(indices, torch.IntTensor(wordsidx))
	 end
      end
      filename = handle:read()
      if _break then print("break 1"); break end
   end
   handle:close()
   print("nb line " .. #indices)
   collectgarbage()
   return {raw=lines, idx=indices, sent=sentences}
end


local function loadwords_back(filename, hash, addraw, feature, maxload)
   print(string.format('loading <%s>', filename))
   local lines = addraw and {} or nil
   local indices = {}
   local sentences = {}
   for line in io.lines(filename) do
      local l = line:gsub(" +", " ")
      table.insert(sentences, l)
      if line~="" then
	 if maxload and maxload > 0 and maxload == #indices then
	    print("breakdata10")
	    break
	 end
	 local words = {}
	 local wordsidx = {}
	 for word in line:gmatch('(%S+)') do
	    if addraw then
	       table.insert(words, word)
	    end
	    table.insert(wordsidx, hash[feature and feature(word) or word] or hash.UNK)
	 end
	 if addraw then
	    table.insert(lines, words)
	 end
	 
	 table.insert(indices, torch.Tensor(wordsidx))
      end
   end

   --print("nb line " .. #indices)

   --print(lines)
   
   collectgarbage()
   return {raw=lines, idx=indices, sent=sentences}
end

local function idx(tbl)
   setmetatable(tbl, {__index = function(self, idx)
			 return self.idx[idx]
   end})
end

local function pad(tbl, sz, val)
   setmetatable(tbl, {__index = function(self, idx)
			 local x = self.idx[idx]
			 local px = torch.Tensor(x:size(1)+2*sz):fill(val)
			 px:narrow(1, sz+1, x:size(1)):copy(x)
			 return px
   end})
end

local function loadentities(pathdata, extention, params)
   local entities = {}
   
   local handle = io.popen("find " .. pathdata .. " -name '*" .. extention .. "' | sort")
   local filename = handle:read()
   while filename do
      --print(string.format('loading caps for <%s>', filename))
      if params.maxload and params.maxload > 0 and params.maxload == #entities then
	 break
      end

      local ent = {}
      for line in io.lines(filename) do
	 --print(line)
	 if line:match("^T%d") then
	    local _ent = line:match("^(T%d+)")
	    local _w2 = line:match("^T%d+\t[^ ]+ %d+[^\t]+%d+\t(.*)")
	    local _type = line:match("^T%d+\t([^ ]+) %d+ %d+")
	    local bounds = line:match("^T%d+\t[^ ]+ (%d+[^\t]+%d+)")
	    local _bounds = {}
	    for b,e in bounds:gmatch("(%d+) (%d+)") do
	       table.insert(_bounds, {tonumber(b),tonumber(e)})
	    end
	    -- local _start = line:match("^T%d+\t[^ ]+ (%d+)")
	    -- local _end = line:match("^T%d+\t[^ ]+ %d+ (%d+)")

	    table.insert(ent, {_bounds, _type, _ent, _w2})
	 end
      end
      -- print(ent)
      -- io.read()
      table.insert(entities, ent)

      filename = handle:read()
   end


   local pad = (params.wsz-1)/2
   local res = torch.Tensor()
   entities.getent = function(data, nsent, e1, e2)
      res:resize(data.words[nsent]:size(1)):fill(2)--2=Other
      for i=1,pad do
	 res[i]=1 --1=Padding
	 res[res:size(1)-i+1] = 1 --1=Padding
      end
      --printw(data.words[nsent], data.wordhash)
      --print(nsent)
      --print(data.words[nsent])
      --print(data.entities[nsent])
      local ent1 = data.entities[nsent][e1][5]
      local ent2 = data.entities[nsent][e2][5]
      for i=1,ent1:size(1) do res[ ent1[i] + pad ]=3 end--entity1
      for i=1,ent2:size(1) do res[ ent2[i] + pad ]=4 end--entity2
      return res
   end
   
   local res2 = torch.Tensor()
   entities.getenttags = function(data, nsent, e1, e2)
      res2:resize(data.words[nsent]:size(1)):fill(data.entityhash["O"])--create input tensor
      for i=1,pad do
	 res2[i]=data.entityhash["PADDING"] --1=Padding
	 res2[res2:size(1)-i+1] = data.entityhash["PADDING"] --1=Padding
      end
      local _type1 = data.entities[nsent][e1][2]
      local _type2 = data.entities[nsent][e2][2]
      local ent1 = data.entities[nsent][e1][5]
      local ent2 = data.entities[nsent][e2][5]
      for i=1,ent1:size(1) do res2[ ent1[i] + pad ] = data.entityhash[_type1] end--entity1
      for i=1,ent2:size(1) do res2[ ent2[i] + pad ] = data.entityhash[_type2] end--entity2
      return res2
   end

   
   entities.nent = function(data, nsent)
      return #data.entities[nsent]
   end

   entities.typeent = function(data, nsent, nent)
      return data.entities[nsent][nent][1]
   end
   
   return entities
end

local function loadentities_back(filename, sents, words, hash, maxload, wsz)
   print(string.format('loading <%s>', filename))
   
   local entities = {}
   local count=0
   --local countprint = 267
   for line in io.lines(filename) do
      --print(count .. " " .. line)
      count = count + 1
      entities[count] = {}
      --print(#entities+1)
      --print(sents[#entities+1])
      --print(line)
      if line~="" then
	 if countprint and count==countprint then print(line); print(sents[count]); io.read() end
	 if maxload and maxload > 0 and #entities>maxload  then
	    print("break entities")
	    break
	 else
	    entities[count] = {}
	    for entity in line:gmatch("[^ %d]+ [%d ]+") do
	       --load boundaries
	       local _type = entity:match("([^ ]+)")
	       local bounds = {}
	       --print(_type)
	       for bound in entity:gmatch("%d+ %d+") do
		  local i1 = bound:match("(%d+) %d+")
		  local i2 = bound:match("%d+ (%d+)")
		  i1 = tonumber(i1)
		  i2 = tonumber(i2)
		  --print(i1)
		  --print(i2)
		  table.insert(bounds, {i1,i2})
	       end
	       --print(bounds)
	       --print(sents[count])

	       if countprint and count==countprint then print(bounds); io.read()end

	       --get the corresponding words
	       local words = {}
	       local enti=1
	       local id = 0
	       local idw = 1
	       local boolent = false
	       if id>=bounds[enti][1] then --sentence start with a drug
		  table.insert(words, idw)
		  boolent = true
		  --print(idw)
	       end
	       --print(#sents)
	       for i=1,#sents[count] do
		  if countprint and count==countprint then print(sents[count]:sub(i,i) .. " " .. i .. " " .. id) end
		  if sents[count]:sub(i,i)~=" " then
		     --if countprint and count==countprint then print("hop") end
		     if id>=bounds[enti][2] then
			if countprint and count==countprint then print("end") end
			boolent = false
			enti=enti+1
			if not bounds[enti] then break end --entity entirely found
		     end
		     id = id + 1 
		  else
		     idw = idw + 1
		     if boolent==false and id>bounds[enti][1] then--start entity when it start in the middle of a word (the cutted entity is included)
			if countprint and count==countprint then print("anomaly") end
			table.insert(words, idw-1)--print(idw);
			boolent=true
		     end
		     if id==bounds[enti][1] then
			if countprint and count==countprint then print("start") end
			--table.insert(words, idw)--print(idw);
			boolent=true
		     end
		     if boolent then table.insert(words, idw) end--print(idw) end
		     --print("====" .. id .. " " .. bounds[enti][2])
		  end
	       end
	       if countprint and count==countprint then print("="); print(words);print("="); io.read() end
	       --if #words==0 then io.read() end
	       table.insert(entities[count], {_type, torch.IntTensor(words)})
	       --print(words)
	       --if #bounds>1 then io.read() end
	       --table.insert(entities[count], {_type, bounds})
	    end
	 end
      end
      --print(entities[count])
   end
   
   local pad = (wsz-1)/2
   local res = torch.Tensor()
   entities.getent = function(data, nsent, e1, e2)
      res:resize(data.words[nsent]:size(1)):fill(2)--2=Other
      for i=1,pad do
	 res[i]=1 --1=Padding
	 res[res:size(1)-pad+1] = 1 --1=Padding
      end
      --printw(data.words[nsent], data.wordhash)
      --print(nsent)
      --print(data.words[nsent])
      --print(data.entities[nsent])
      local ent1 = data.entities[nsent][e1][2]
      local ent2 = data.entities[nsent][e2][2]
      for i=1,ent1:size(1) do res[ ent1[i] + pad ]=3 end--entity1
      for i=1,ent2:size(1) do res[ ent2[i] + pad ]=4 end--entity2
      return res
   end
   
   local res2 = torch.Tensor()
   entities.getenttags = function(data, nsent)
      res2:resize(data.words[nsent]:size(1) + (2*pad)):fill(hash["O"])--create input tensor
      for i=1,pad do
	 res2[i]=hash["PADDING"] --1=Padding
	 res2[res2:size(1)-pad+1] = hash["PADDING"] --1=Padding
      end
      for i=1,#data.entities[nsent] do
	 local _type = data.entities[nsent][i][1]
	 for j=1,data.entities[nsent][i][2]:size(1) do
	    res2[data.entities[nsent][i][2][j] + pad] = hash[_type]
	 end
      end
      return res2
   end

   entities.nent = function(data, nsent)
      return #data.entities[nsent]
   end

   entities.typeent = function(data, nsent, nent)
      return data.entities[nsent][nent][1]
   end
   
   return entities
   
end

function loadrelations_back(filename, hash, maxload, params)
   print(string.format('loading <%s>', filename))
   local distribution = torch.Tensor(#hash):fill(0)
   local relations = {}
   local count = 0
   for line in io.lines(filename) do
      count = count + 1
      relations[count] = {}
      --print(#entities+1)
      --print(sents[#entities+1])
      if not line:match("^[\t ]*$") then
	 if maxload and maxload > 0 and #relations>maxload then
	    print("break relations")
	    break
	 else
	    --print(line)
	    for relation in line:gmatch("%d+ %d+ [^%d ]+") do
	       --print(relation)
	       local e1 = relation:match("%d+")
	       local e2 = relation:match("%d+ (%d+)")
	       local _type = relation:match("%d+ %d+ ([^%d ]+)")
	       e1 = tonumber(e1)+1
	       e2 = tonumber(e2)+1
	       --print(e1)
	       --print(e2)
	       if relations[count][e1]==nil then relations[count][e1]={} end
	       relations[count][e1][e2] = hash[_type]--, {_type, e2})
	       distribution[hash[_type]] = distribution[hash[_type]] + 1
	    end
	 end
      end
   end
   
   relations.isrelated = function(self, nsent, e1, e2)
      --print(self[nsent])
      --print(nsent)
      --print(e1)
      --print(e2)
      return self[nsent][e1] and self[nsent][e1][e2] or hash["null"] 
   end

   -- for i=1,distribution:size(1) do
   --    distribution[i] = distribution[i]==0 and 0 or ( 1 / distribution[i])
   -- end
   --distribution:norm()
   --distribution = distribution / distribution:max()
   --print(distribution)
   local min = math.huge
   for i=1,distribution:size(1) do
      if distribution[i]~=0 and distribution[i]<min then min = distribution[i] end
   end
   distribution = distribution / min
   --print(distribution)
   relations.distribution = distribution
   
   return relations
end

local function loadrelations(pathdata, extention, maxload, hash)
   local relations = {}
   local count = 0

   local handle = io.popen("find " .. pathdata .. " -name '*" .. extention .. "' | sort")
   local filename = handle:read()
   while filename do
      count = count + 1
      relations[count] = {}
      
      if maxload and maxload > 0 and #relations>maxload then
	 break
      end

      local rel = {}
      for line in io.lines(filename) do
	 if line:match("^R%d") then
	    local ent1 = line:match("^R%d+\t[^ ]+ Arg1:T(%d+)")
	    local ent2 = line:match("^R%d+\t[^ ]+ Arg1:T%d+ Arg2:T(%d+)")
	    local _type = line:match("^R%d+\t([^ ]+) Arg1:T%d+ Arg2:T%d+")
	    ent1 = tonumber(ent1)
	    ent2 = tonumber(ent2)
	    if relations[count][ent1]==nil then relations[count][ent1]={} end
	    relations[count][ent1][ent2] = hash[_type]--, {_type, e2})
	    if relations[count][ent2]==nil then relations[count][ent2]={} end
	    relations[count][ent2][ent1] = hash[_type]--, {_type, e2})

	    table.insert(rel, {ent1, ent2})
	 end
      end

      filename = handle:read()
   end

   relations.isrelated = function(self, nsent, e1, e2)
      assert(e1<e2)
      --print(self[nsent])
      --print(nsent)
      --print(e1)
      --print(e2)
      return self[nsent][e1] and self[nsent][e1][e2] or hash["null"] 
   end

   return relations
end



local wordhash, entityhash, deptypehash, poshash, relationhash

function loadhash(params)

   local path = "data/"
   
   wordhash = wordhash or _loadhash('data/hash/word.txt', params.nword)
   entityhash = entityhash or _loadhash('data/hash/entities.txt')
   relationhash = relationhash or _loadhash("data/hash/relations.txt")
   
end

function createdata(params)
   
   local pathdata = params.data
   
   local words = loadwords(pathdata, wordhash, params.addraw, wordfeature, params.maxload)
   pad(words, (params.wsz-1)/2, wordhash.PADDING)
   
   local starts, ends = loadstartend(pathdata, nil, params.maxload)
   
   local entities = loadentities(pathdata, ".ann",  params)
   load_entity_indices(entities, words, starts, ends, wordhash)
   
   loaddag(entities)
   
   local relations = loadrelations(pathdata, ".ann", params.maxload, relationhash)
   
   return {wordhash=wordhash, entityhash=entityhash, relationhash=relationhash, words=words, entities=entities, relations=relations, size=#words.idx}
   
end


function extract_data(data, percentage, sector, remove)

   remove = remove or false
   print("Extracting data. Remove=" .. (remove and "true" or "false"))
   local size = data.size

   local subsize = math.floor((size*percentage)/100)

   local start = (subsize * (sector-1))+1

   print("\tsize: " .. size .. " subcorpus size: " .. subsize .. " subcorpus start at " .. start)

   local tabs = {words=true, pos=true}

   local new_size_expected = size - subsize
   
   local newdata = {}
   for k,v in pairs(tabs) do
      if data[k] then
	 local newtab = {}
	 for i=1,subsize do
	    table.insert(newtab, data[k].idx[remove and start or (start+i-1)])
	    if remove then table.remove(data[k].idx, start) end
	    if remove then table.remove(data[k].sent, start) end
	 end
	 newdata[k] = {idx=newtab}
	 setmetatable(newdata[k], getmetatable(data[k]))
      end
   end

   local tabs = {entities=true,relations=true,ids=true}
   --local tabs = {words=true}
   
   -- if false then
   --    print("data")
   --    for i=1, 85 do
   -- 	 io.write(#data.entities[i] .. " ")
   --    end
   --    io.write("\n")
   -- end

   for k,v in pairs(tabs) do
      if data[k] then
	 local newtab = {}
	 for i=1,subsize do
	    table.insert(newtab, data[k][remove and start or (start+i-1)])
	    if remove then table.remove(data[k], start) end
	 end
	 newdata[k] = newtab
      end
   end
   
   newdata.entities.nent = data.entities.nent
   newdata.entities.typeent = data.entities.typeent
   newdata.entities.getent = data.entities.getent
   newdata.entities.getenttags = data.entities.getenttags

   newdata.relations.isrelated = data.relations.isrelated
   
   data.size = #data.words.idx
   newdata.size = #newdata.words.idx

   if remove then assert(data.size==new_size_expected, size .. " " .. data.size .. " " .. new_size_expected) end
   if remove then assert(newdata.size==subsize) end
   
   --print("====")
   for k,v in pairs(data) do
      if not newdata[k] then newdata[k] = data[k] end
   end
   
   -- if false then
   --    print("newdata")
   --    for i=1, 35 do
   -- 	 io.write("  ")
   --    end
   --    io.write(" ")
   --    for i=1, 35 do
   -- 	 io.write(#newdata.entities[i] .. " ")
   --    end
   --    io.write("\n")
   
   --    print("olddata")
   --    for i=1, 35 do
   -- 	 io.write(#data.entities[i] .. " ")
   --    end
   --    for i=1, 35 do
   -- 	 io.write("  ")
   --    end
   
   --    for i=36, 50 do
   -- 	 io.write(#data.entities[i] .. " ")
   --    end
   --    io.write("\n")
   -- end

   return newdata
   
end
