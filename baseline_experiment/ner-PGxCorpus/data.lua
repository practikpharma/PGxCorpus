require 'torch'

function clean_entities(entities)
   for i=1,#entities do
      entities[i].level=nil
      entities[i].sons=nil
   end
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
	 local b1, e1 = ent1[i][1], ent1[i][2]
	 local b2, e2 = ent2[j][1], ent2[j][2]
	 if type(b1)=="string" then error("") end
	 if b2<=b1 and e2>=e1 then found = true end
	 if b1==b2 and e1==e2 then found = false end --only for test if the same entity is predicted twice
      end
      res = res and found
   end
   -- print("is_included")
   -- print(res)
   return res
end

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

function _printdag(ent, ind)
   --if ent[3]=="T6" and ent[4]=="CpG immunomer" then io.read() end
   --print(ent)
   print(ind .. " - " .. ent[3] .. " (" .. ent.level .. ")")
   for i=1,#ent.sons do
      _printdag(ent.sons[i], ind .. "\t")
   end
end

function printdag(entities)
   for i=1,#entities do
      print(i .. " " .. entities[i][3])
      print(entities[i][1])
      print(entities[i][4])
   end
   for i=1,#entities do
      _printdag(entities[i], "")
   end
end

local function loadnames(pathdata, maxload)
   print("loading names in " .. pathdata)
   local indices = {}
   local handle = io.popen("find " .. pathdata .. " -name '*.txt' | sort")
   
   local filename = handle:read()
   while filename do
      --print(string.format('loading <%s>', filename))
      
      if maxload and maxload > 0 and maxload == #indices then
	 print("breakdata1")
	 break
      end
      table.insert(indices, filename:match("/(%d+_%d+).txt"))
      filename = handle:read()
   end
   handle:close()
   collectgarbage()
   return indices
end

function indice_to_bioes(t, indices, chunkhash, pad, tag)
   if #indices==1 then
      assert(t[ indices[1] + pad ] == chunkhash["O"], "Damn it!")
      t[ indices[1] + pad ] = chunkhash["S-" .. tag]
   else
      assert(t[ indices[1] + pad ] == chunkhash["O"], "Damn it!")
      t[ indices[1] + pad ] = chunkhash["B-" .. tag]
      for i=2,#indices-1 do
	 assert(t[ indices[i] + pad ] == chunkhash["O"], "Damn it!")
	 t[ indices[i] +pad] = chunkhash["I-" .. tag]
      end
      assert(t[ indices[ #indices ] + pad ] == chunkhash["O"], "Damn it!")
      t[ indices[ #indices ] + pad ] = chunkhash["E-" .. tag]
   end
end

--same but without testing if the location is empty ("O")
function indice_to_bioes2(t, indices, chunkhash, pad, tag)
   if #indices==1 then
      t[ indices[1] + pad ] = chunkhash["S-" .. tag]
   else
      t[ indices[1] + pad ] = chunkhash["B-" .. tag]
      for i=2,#indices-1 do
	 t[ indices[i] +pad] = chunkhash["I-" .. tag]
      end
      t[ indices[ #indices ] + pad ] = chunkhash["E-" .. tag]
   end
end


--extraction prediction labels
function extract_pred_labels(entities, words, pad, chunkhash)
   local indices = {}
   for i=1,#entities do --for each sentence
      --print("======================")
      --print(words.sent[i])
      --printdag(entities[i], "")
      
      --getting max level
      local level_max = 0
      for j=1,#entities[i] do
	 level_max = math.max(level_max, entities[i][j].level)
      end
      --print("level max: " .. level_max)

      local levels = {}
      for l=1,level_max do
	 --print("level_" .. l)
	 --print(chunkhash["O"])
	 local t = torch.Tensor(words[i]:size(1)):fill( chunkhash["O"]);
	 for p=1, pad do
	    t[p] = chunkhash["PADDING"]
	    t[ t:size(1)-p+1 ] = chunkhash["PADDING"]
	 end
	 for e=1,#entities[i] do
	    local level = entities[i][e].level 
	    local tag = entities[i][e][2]
	    --print(tag)
	    if level==l then
	       indice_to_bioes(t, entities[i][e][5], chunkhash, pad, tag)
	       -- assert(#entities[i][e][1]==1, "discontiguous?: not yet")
	       -- print(entities[i][e])
	       -- t[ entities[i][e][5][1] + pad ] = chunkhash["B-" .. tag]
	       -- for k=2,#entities[i][e][5]-1 do
	       -- 	  t[ entities[i][e][5][k] +pad] = 1
	       -- end
	       --t[ entities[i][e][5][ #entities[i][e] ] +pad] = chunkhash["B-" .. tag]
	    end
	 end
	 table.insert(levels, t)
	 --printw(t, chunkhash)
      end
      local t = torch.Tensor(words[i]:size(1)):fill( chunkhash["O"]);
      table.insert(levels, t) --final prediction filled only with Other
	 
      indices[i] = levels
      --if words.sent[i]:match("The decrement for daily dosage of Tac") then io.read() end

      --io.read()
   end
   --print("end extract_label")
      
   return indices
end


function _extract_input_labels(ents, words, pad, labelhash)
   --print("======================")
   --print(words.sent[i])
   --printdag(ents, "")
   
   --getting max level
   local level_max = 0
   for j=1,#ents do
      level_max = math.max(level_max, ents[j].level)
   end
   --print("level max: " .. level_max)
   
   local levels = {}
   for l=0,level_max do
      --print("level_" .. l)
      --print(labelhash["O"])
      local t = torch.Tensor(words:size(1)):fill( labelhash["O"]);
      for p=1, pad do
	 t[p] = labelhash["PADDING"]
	 t[ t:size(1)-p+1 ] = labelhash["PADDING"]
      end
      
      
      for lev=1,l do
	 for e=1,#ents do
	    local level = ents[e].level 
	    local tag = ents[e][2]
	    --print(tag)
	    if level==lev then
	       indice_to_bioes2(t, ents[e][5], labelhash, pad, tag)
	    end
	 end
      end
      
      table.insert(levels, t)
      --printw(t, labelhash)
   end
   return levels
end
   
--extraction input labels
function extract_input_labels(entities, words, pad, labelhash)
   local indices = {}
   for i=1,#entities do --for each sentence

      local levels = _extract_input_labels(entities[i], words[i], pad, labelhash)
      indices[i] = levels

   end
   --print("end extract_label")
      
   return indices
end




function caps(word)
   local caps
   if word:match("^[%l%A]*$")~=nil then caps=1
   elseif word:match("^[%u%A]*$")~=nil then caps=2
   elseif word:match("^%u")~=nil then caps=3
   elseif word:match("^%U.*%u")~=nil then caps=4
   else error("")
   end
   return caps
end

local function loadhash(filename, maxidx)
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

local function wordfeature(word)
   word = word:lower()
   word = word:gsub('%d+', '0')
   return word
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

local function loadstartend(pathdata, feature, maxload)
   print("loading startend in " .. pathdata)
   local starts = {}
   local ends = {}

   local handle = io.popen("find " .. pathdata .. " -name '*.txt' | sort")
   local filename = handle:read()
   while filename do
      --print(string.format('loading startend for <%s>', filename))
      for line in io.lines(filename) do
	 --print(line)
	 if maxload and maxload > 0 and maxload == #starts then break end
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
   end
   handle:close()
   return starts, ends
end

local function loadentities(pathdata, extention, maxload)
   local indices = {}
   
   local handle = io.popen("find " .. pathdata .. " -name '*" .. extention .. "' | sort")
   local filename = handle:read()
   while filename do
      --print(string.format('loading caps for <%s>', filename))
      if maxload and maxload > 0 and maxload == #indices then
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
      table.insert(indices, ent)

      filename = handle:read()
   end
   return indices
end

function loaddag(entities)
   local indices = {}
   for i=1,#entities do
      --print("------------------------------------------------> " .. i)
      -- print("<<<<<")
      -- print(entities[i][1])
      -- print(entities[i][2])
      -- print(">>>>>")
      getdag(entities[i])
      -- print("<<<<<")
      -- print(entities[i][1])
      -- print(entities[i][2])
      -- print(">>>>>")
      -- io.read()
      setlevel(entities[i], true)
      --printdag(entities[i])
      -- print(entities[i])
      --io.read()
   end
   return indices
end

function levelmax(entities, level)
   for e = 1,#entities do
      local i = 1
      while entities[e][i] do
	 if entities[e][i].level>level then table.remove(entities[e], i) else i = i + 1 end
      end
   end
end

function onlylabel(entities, ent)
   for e=1, #entities do
      local i = 1
      while entities[e][i] do
	 if not ent[entities[e][i][2]] then table.remove(entities[e], i) else i = i + 1 end
      end
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
      
      ents[j][5] = idx
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



local function loadparses(filename, hash, addraw, maxload)
   print(string.format('loading <%s>', filename))
   local lines = addraw and {} or nil
   local indices = {}
   for line in io.lines(filename) do
      if maxload and maxload > 0 and maxload == #indices then
         break
      end

      local words = {}
      local wordsidx = {}
      for word,size in line:gmatch('(%S+) (%S+)') do
	 if addraw then
            table.insert(words, word)
         end
         table.insert(wordsidx, {size,hash[word]})
      end
      if addraw then
         table.insert(lines, words)
      end
      table.insert(indices, torch.IntTensor(wordsidx))
   end
   collectgarbage()
   return {raw=lines, idx=indices}
end

local function loadparsesbioes(filename, hash, addraw, maxload)
   print(string.format('loading <%s>', filename))
   local lines = addraw and {} or nil
   local indices = {}
   for line in io.lines(filename) do
      if maxload and maxload > 0 and maxload == #indices then
         break
      end
      --print(line)
      local words = {}
      local wordsidx = {}
      for word,size in line:gmatch('(%S+) (%S+)') do
	 size = tonumber(size)
         if addraw then
            table.insert(words, word)
         end
         if word == 'O' then
            assert(size == 1)
            table.insert(wordsidx, hash[word])
         elseif size == 1 then
	    if not hash["S-" .. word] then
	       print(word)
	       io.read()
	    end
	    table.insert(wordsidx, hash['S-' .. word])
	 else            
            if not hash["B-" .. word] then
	       print(word)
	       io.read()
	    end
	    table.insert(wordsidx, hash['B-' .. word])
            for i=1,size-2 do
               table.insert(wordsidx, hash['I-' .. word])
            end
            table.insert(wordsidx, hash['E-' .. word])
         end
      end
      if addraw then
         table.insert(lines, words)
      end
      table.insert(indices, torch.IntTensor(wordsidx))
   end
   collectgarbage()
   return {raw=lines, idx=indices}
end


local function idx(tbl)
   setmetatable(tbl, {__index = function(self, idx)
                                   return self.idx[idx]
                                end})
end

local function pad(tbl, sz, val)
   setmetatable(tbl, {__index = function(self, idx)
			 local x = self.idx[idx]
			 local px = torch.IntTensor(x:size(1)+2*sz):fill(val)
			 px:narrow(1, sz+1, x:size(1)):copy(x)
			 return px
                                end})
end


local wordhash
local wordhashupper
local taghash
local chunkhash
local pubtatorhash
local marmothash
local marmothashes
local level1hash

function createdata(params)
   local data = {}

   local path = "data/"
   local pathdata = params.data
   
   wordhash = wordhash or loadhash('data/hash/word.txt', params.nword)
   chunkhash = chunkhash or loadhash('data/hash/' .. params.task .. '-bioes.txt')
   labelhash = labelhash or loadhash('data/hash/' .. params.task .. '-bioes-pad.txt')
   entityhash = entityhash or loadhash('data/hash/' .. params.task .. '.txt')
   pubtatorhash = pubtatorhash or loadhash('data/hash/pubtator-bioes.txt')

   -- if params.hierarchy==1 then
   --    local tab = {Phenotype="Phenotype", Disease="Phenotype", Pharmacokinetic_phenotype="Phenotype", Pharmacodynamic_phenotype="Phenotype", Genomic_factor="Genomic_factor", Genomic_variation="Genomic_factor", Gene_or_protein="Genomic_factor", Limited_variation="Genomic_factor", Haplotype="Genomic_factor", Chemical="Chemical"}
   --    for i=1,#chunkhash-1 do -- -1 for other
   -- 	 local bies = chunkhash[i]:match("([BIES])%-")
   -- 	 local tag = chunkhash[i]:match("[BIES]%-(.*)")
   -- 	 chunkhash[ chunkhash[i] ] = chunkhash[ bies .. "-" .. tab[tag] ]
   --    end
   -- elseif params.hierarchy==2 then
   --    local tab = {Phenotype="Phenotype", Disease="Phenotype", Pharmacokinetic_phenotype="Phenotype", Pharmacodynamic_phenotype="Phenotype", Genomic_factor="Genomic_variation", Genomic_variation="Genomic_variation", Gene_or_protein="Gene_or_protein", Limited_variation="Genomic_variation", Haplotype="Genomic_variation", Chemical="Chemical"}
   --    for i=1,#chunkhash-1 do -- -1 for other
   -- 	 local bies = chunkhash[i]:match("([BIES])%-")
   -- 	 local tag = chunkhash[i]:match("[BIES]%-(.*)")
   -- 	 chunkhash[ chunkhash[i] ] = chunkhash[ bies .. "-" .. tab[tag] ]
   --    end
   -- end
   
   
   local words, caps, tags, chunks, marmottags, starts, ends, chars, putators, level1

   local file
   if params.task=="pgx" then
      file = pathdata .. "/words.txt" -- notejo: is it still usefull?
   else
      error("unknown task")
   end
   
   words  = loadwords(pathdata, wordhash, params.addraw, wordfeature, params.maxload)
   pad(words, (params.wsz-1)/2, wordhash.PADDING)

   starts, ends = loadstartend(pathdata, nil, params.maxload)

   local names = loadnames(pathdata, params.maxload)
   
   local entities = loadentities(pathdata, ".ann",  params.maxload)
   onlylabel(entities, params.onlylabel)
   loaddag(entities)
   if params.levelmax~=math.huge then levelmax(entities, params.levelmax) end
   
   load_entity_indices(entities, words, starts, ends, wordhash)
   local labels = extract_pred_labels(entities, words, (params.wsz-1)/2, labelhash)
   local labels_input = extract_input_labels(entities, words, (params.wsz-1)/2, labelhash)
   
   local entities_pubtator = loadentities(pathdata, ".ann_pubtator",  params.maxload)
   loaddag(entities_pubtator)
   load_entity_indices(entities_pubtator, words, starts, ends, wordhash)
   local labels_pubtator = extract_pred_labels(entities_pubtator, words, (params.wsz-1)/2, pubtatorhash)
   
   for i=1,#labels_pubtator do
      assert(#labels_pubtator[i]==2) --pubtator entries + 
   end
   
   return {entityhash=entityhash, entities=entities, labels=labels, labels_input=labels_input, names=names, wordsupper=wordsupper, genedict=genedict, words=words, caps=caps, tags=tags, chunks=chunks, wordhash=wordhash, chunkhash=chunkhash, labelhash=labelhash, taghash=taghash, size=#words.idx, marmothashes=marmothashes, marmottags=marmottags, marmothash=marmothash, starts=starts, ends=ends, labels_pubtator=labels_pubtator, pubtatorhash=pubtatorhash, level1=level1, level1hash=level1hash, getdag=getdag, setlevel=setlevel, _extract_input_labels=_extract_input_labels, _load_entity_indices=_load_entity_indices, clean_entities=clean_entities}
end



function extract_data(data, percentage, sector)
   print("Extracting data")
   local size = data.size
   
   local subsize = math.floor((size*percentage)/100)
   
   local start = (subsize * (sector-1))+1

   if start+subsize>size then error("only " .. math.floor(size/subsize) .. " subcorpora available usinng " .. percentage .. " % of the corpus. You chose sector " .. sector .. "...") end
   print("\tsize: " .. size .. " subcorpus size: " .. subsize .. " subcorpus start at " .. start)

   local tabs = {words=true, caps=true}
   --, starts=true, ends=true

   local new_size_expected = size - subsize
   
   local newdata = {}
   for k,v in pairs(tabs) do
      if data[k] then
	 local newtab = {}
	 for i=1,subsize do
	    table.insert(newtab, data[k].idx[start])
	    table.remove(data[k].idx, start)
	    --table.remove(data[k].sent, start)
	 end
	 local newtabsent = {}
	 if k=="words" or k=="pubtators" then
	    for i=1,subsize do
	       table.insert(newtabsent, data[k].sent[start])
	       table.remove(data[k].sent, start)
	       --table.remove(data[k].sent, start)
	    end
	 end
	 newdata[k] = {idx=newtab, sent=newtabsent}
	 setmetatable(newdata[k], getmetatable(data[k]))
      end
   end


   
   local tabs = {starts=true, ends=true, names=true, labels=true, labels_input=true, entities=true, labels_pubtator=true}
   -- local tabs = {trees2=true,trees=true,entities=true,relations=true,ids=true, select_data=true}
   -- --local tabs = {words=true}
   
   -- -- if false then
   -- --    print("data")
   -- --    for i=1, 85 do
   -- -- 	 io.write(#data.entities[i] .. " ")
   -- --    end
   -- --    io.write("\n")
   -- -- end

   for k,v in pairs(tabs) do
      if data[k] then
	 --print(k)
   	 local newtab = {}
   	 for i=1,subsize do
   	    table.insert(newtab, data[k][start])
   	    table.remove(data[k], start)
   	 end
   	 newdata[k] = newtab
      end
   end
   
   data.size = #data.words.idx
   newdata.size = #newdata.words.idx

   assert(data.size==new_size_expected, size .. " " .. data.size .. " " .. new_size_expected)
   assert(newdata.size==subsize)
   
   --print("====")
   for k,v in pairs(data) do
      if not newdata[k] then newdata[k] = data[k] end
   end

   return newdata
   
end
