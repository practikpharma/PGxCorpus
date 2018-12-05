local graph = require("HopcroftKarp")

function match(e1, e2)
   --same number of entities?
   if #e1~=#e2 then return false end
   
   for i=1,#e1 do
      if e1[i][1]~=e2[i][1] or e1[i][2]~=e2[i][2] then return false end
   end

   return true
end

function match2(e1, e2) --partial match (any overlap)
   for i=1,#e1 do
      for j=1,#e2 do
	 local a1, a2, b1, b2 = e1[i][1], e1[i][2], e2[j][1], e2[j][2]
	 if (a1>=b1 and a1<=b2) or (a2>=b1 and a2<=b2) or (b1>=a1 and b1<=a2) or (b2>=a1 and b2<=a2) then return true end 
      end
   end
   return false
end

function repair(e1)
   table.sort(e1, function(a,b) return a[1]<b[1] end)--just in case
   local i=1
   while i<#e1 do
      if e1[i+1] and (e1[i][2]==e1[i+1][1] or e1[i][2]+1==e1[i+1][1]) then
	 e1[i][2] = e1[i+1][2]
	 table.remove(e1, i+1)
	 --print("=======================================================> repaired")
      else
	 i = i+1
      end
   end
   return e1
end

   
function lbl2chunks(lbl)
   local posw = 1
   local pos = 1
   local res = {}
   while pos<#lbl do
      --print(lbl[pos])
      if lbl[pos]=="O" then
	 posw = posw + 1
      else
	 table.insert(res, {lbl[pos+1], lbl[pos], posw})
	 posw = posw + lbl[pos+1]
      end
      pos = pos + 2
   end
   return res
end

function lbl2tags(lbl)
   local current = ""
   local pos = 1
   local res = {}
   while pos<#lbl do
      if lbl[pos]=="O" or lbl[pos]~=current then
	 if lbl[pos]=="O" then
	    table.insert(res, "O")
	    for i=1, lbl[pos+1]-1 do
	       table.insert(res, "O")
	       io.write("???")
	    end
	    current = lbl[pos]
	    pos = pos + 2
	 else	 
	    table.insert(res, "I-" .. lbl[pos])
	    for i=1, lbl[pos+1]-1 do
	       table.insert(res, "I-" .. lbl[pos])
	    end
	    current = lbl[pos]
	    pos = pos + 2
	 end
      else
	 table.insert(res, "B-" .. lbl[pos])
	 for i=1, lbl[pos+1]-1 do
	    table.insert(res, "I-" .. lbl[pos])
	 end
	 current = lbl[pos]
	 pos = pos + 2
      end
   end

   return res
end

local function writeinfile(tabstartend, fann, indice, suffix)
   suffix = suffix or ""
   for i=1,#tabstartend do
      --print(i)
      fann:write("T" .. (i+indice) .. "\t" .. tabstartend[i][5] .. suffix .. " " .. tabstartend[i][1] + tabstartend[i][4]-1 .. " " .. tabstartend[i][2] + tabstartend[i][4]+ #tabstartend[i][3] - 1 .. "\t")
      for j=1,#tabstartend[i][3] do
	 fann:write(tabstartend[i][3][j] .. " ")
      end
      --fann:write("# " .. tabstartend[i][1] .. " " .. tabstartend[i][2])
      fann:write("\n")
   end
end
      

function gettabstartend(lbl, data, tabwords, idx)
   local tabstartend = {}
   local pos = 1
   local i=1
   while i<=#lbl do
      if lbl[i] =="O" then i=i+2; pos=pos+1
      else 
	 local tabw = {}
	 for k=pos, pos+lbl[i+1]-1 do 
	    table.insert(tabw, tabwords[k])
	 end
	 table.insert(tabstartend, {data.starts[idx][pos], data.ends[idx][pos+lbl[i+1]-1], tabw, pos, lbl[i]})
	 pos = pos + lbl[i+1]
	 i = i + 2
      end
   end
   return tabstartend
end

function path2lbl(path, params, data, hash)
   local tab = {}
   for i=1,path:size(1) do
      z=path[i]
      if z>0 then
	 --print(z)
	 local label = hash[(z-1) % params.nlabel +1]
	 local sz = math.floor((z-1) / params.nlabel) +1
	 table.insert(tab,label)
	 table.insert(tab,sz)
      end
   end
   return tab
end

   
function path2lblbioes(path, params, data, hash)
   --print(data.chunkhash)
   local tab = {}
   local i = 0
   while i < path:size(1) do
      i=i+1
      local z = path[i]
      if z>0 then
	 local label = hash[z]
	 --print(label)
	 local tag = label:gsub('^.%-', '')
	 
	 if label == 'O' then
	    table.insert(tab, 'O')
	    table.insert(tab,1)
	 else
	    local sz = 1
	    while not label:match('^S%-') and not label:match('^E%-') do
	       i=i+1
	       sz = sz + 1
	       label = hash[path[i]]
	    end
	    table.insert(tab, tag)
	    table.insert(tab, sz)
	 end
      end
   end
   return tab
end

function compute_prf1(tab_entities, tab_ent)
   local tab_return = {}
   local avg_p, avg_r = 0,0 --for macro average f1
   for i=1, #tab_ent do
      local k = tab_ent[i]
      local v = tab_entities[k]
      
      -- print("\n%=========================================================================")
      -- print("%==================== " .. k .. " ============================")
      -- print("%=========================================================================")
      -- print(v.ent_total)
      -- print(v.ent_tp)
      -- print(v.ent_fp)
      
      local tpfp = v.ent_tp + v.ent_fp
      local precision = v.ent_tp / tpfp
      
      -- print("============Precision==============")
      -- print(precision)
	 
      local recall = v.ent_tp / v.ent_total
      
      -- print("============Recall==============")
      -- print(recall)
      
      local prod = precision * recall
      local add = precision + recall
      local f1 = (2*prod) / add
      
      -- print("============F1==============")
      -- print(f1)
	 
      tab_return[k] = {precision=precision, recall=recall, f1=f1, ent_total=v.ent_total, ent_tp=v.ent_tp, ent_fp=v.ent_fp}

      avg_p = avg_p + (precision==precision and precision or 0)
      avg_r = avg_r + (recall==recall and recall or 0)
   end

   avg_p = avg_p / #tab_ent
   avg_r = avg_r / #tab_ent

   tab_return.macro_avg = {precision=avg_p, recall=avg_r, f1=(2*avg_p*avg_r)/(avg_p+avg_r)}
   return tab_return
end


function compute_prf1_2(tab_entities, tab_ent)
   local tab_return = {}
   local avg_p, avg_r = 0,0 --for macro average f1
   for i=1, #tab_ent do
      local k = tab_ent[i]
      local v = tab_entities[k]
      
      -- print("\n%=========================================================================")
      -- print("%==================== " .. k .. " ============================")
      -- print("%=========================================================================")
      -- print(v.ent_tp)
      -- print(v.ent_fp)
      -- print(v.ent_fn)
      
      local tpfp = v.ent_tp + v.ent_fp
      local precision = tpfp==0 and 1 or (v.ent_tp / tpfp)
      
      -- print("============Precision==============")
      -- print(precision)

      local tpfn = v.ent_tp + v.ent_fn
      local recall = tpfn==0 and 1 or  (v.ent_tp / tpfn)
      
      -- print("============Recall==============")
      -- print(recall)
      
      local prod = precision * recall
      local add = precision + recall
      local f1 = (2*prod) / add
      
      -- print("============F1==============")
      -- print(f1)
	 
      tab_return[k] = {precision=precision, recall=recall, f1=f1, ent_fn=v.ent_fn, ent_tp=v.ent_tp, ent_fp=v.ent_fp}

      avg_p = avg_p + (precision==precision and precision or 0)
      avg_r = avg_r + (recall==recall and recall or 0)
   end

   avg_p = avg_p / #tab_ent
   avg_r = avg_r / #tab_ent

   tab_return.macro_avg = {precision=avg_p, recall=avg_r, f1=(2*avg_p*avg_r)/(avg_p+avg_r)}
   return tab_return
end


local tab_ent = {"Chemical", "Genomic_factor", "Limited_variation", "Genomic_variation", "Gene_or_protein", "Haplotype", "Phenotype", "Disease", "Pharmacokinetic_phenotype", "Pharmacodynamic_phenotype"}
local hierarchy_ent = {}
for i=1,#tab_ent do
   hierarchy_ent[tab_ent[i]] = {}
   hierarchy_ent[tab_ent[i]][tab_ent[i]] = true
end
hierarchy_ent["Genomic_factor"]["Gene_or_protein"] = true
hierarchy_ent["Genomic_factor"]["Genomic_variation"] = true
hierarchy_ent["Genomic_factor"]["Haplotype"] = true
hierarchy_ent["Genomic_factor"]["Limited_variation"] = true

hierarchy_ent["Genomic_variation"]["Haplotype"] = true
hierarchy_ent["Genomic_variation"]["Limited_variation"] = true

hierarchy_ent["Phenotype"]["Pharmacokinetic_phenotype"] = true
hierarchy_ent["Phenotype"]["Pharmacodynamic_phenotype"] = true
hierarchy_ent["Phenotype"]["Disease"] = true



local back_hierarchy_ent = {}
back_hierarchy_ent["Haplotype"] = "Genomic_variation"
back_hierarchy_ent["Limited_variation"] = "Genomic_variation"

back_hierarchy_ent["Genomic_variation"] = "Genomic_factor"
back_hierarchy_ent["Gene_or_protein"] = "Genomic_factor"

back_hierarchy_ent["Pharmacokinetic_phenotype"] = "Phenotype"
back_hierarchy_ent["Pharmacodynamic_phenotype"] = "Phenotype"
back_hierarchy_ent["Disease"] = "Phenotype"



function equal_ent(ent1, ent2, consider_hierarchy)
   if consider_hierarchy then
      return (hierarchy_ent[ent1][ent2] or hierarchy_ent[ent2][ent1]) or false
   else
      return ent1==ent2
   end
end



function fill_tab_entities(params, data, tab_entities, prediction, target)
   local verbose = false
   if verbose then print("prediction : " .. (prediction and prediction or "nil")) end
   if verbose then print("target : " .. (target and target or "nil")) end
   if target and hierarchy_ent[target][prediction] then
      if verbose then print("the prediction (" .. prediction .. ") is more specific than the target (" .. target .. ")") end
      local current = prediction
      while current~=target do
	 if verbose then print(current .. " is a false positive 1") end
	 tab_entities[current].ent_fp = tab_entities[current].ent_fp + 1 
	 current = back_hierarchy_ent[current]
	 end
      while current do
	 if verbose then print(current .. " is a true positive 2") end
	 tab_entities[current].ent_tp = tab_entities[current].ent_tp + 1 
	 current = back_hierarchy_ent[current]
      end
   elseif prediction and hierarchy_ent[prediction][target] then
      if verbose then print("the prediction (" .. prediction .. ") is less specific than the target (" .. target .. ")") end
      local current = target
      while current~=prediction do
	 if verbose then print(current .. " is a false negative 3") end
	 tab_entities[current].ent_fn = tab_entities[current].ent_fn + 1 
	 current = back_hierarchy_ent[current]
      end
      while current do
	 if verbose then print(current .. " is a true positive 4") end
	 tab_entities[current].ent_tp = tab_entities[current].ent_tp + 1
	 current = back_hierarchy_ent[current]
      end
   else
      local target_ancestors = {}
      local current = target
      while current do
	 target_ancestors[current]=true
	 current = back_hierarchy_ent[current]
      end
      local current = prediction
      while current do
	 if target_ancestors[current] then
	    if verbose then print(current .. " is a true positive 5") end
	    tab_entities[current].ent_tp = tab_entities[current].ent_tp + 1
	    target_ancestors[current] = false
	 else
	    if verbose then print(current .. " is a false positive 6") end
	    tab_entities[current].ent_fp = tab_entities[current].ent_fp + 1
	 end
	 current = back_hierarchy_ent[current]
      end
      for k,v in pairs(target_ancestors) do
	 if v then
	    if verbose then print(k .. " is a false negative 7") end
	    --confusion_matrix[k_indice][ data.relationhash["null"] ] = confusion_matrix[k_indice][ data.relationhash["null"] ] + 1
	    tab_entities[k].ent_fn = tab_entities[k].ent_fn + 1 
	 end
      end
   end
   if verbose then print("\ntp\tfp\tfn") end
   for k,v in pairs(tab_entities) do
      if verbose then print(v.ent_tp .. "\t" .. v.ent_fp .. "\t" .. v.ent_fn .. "\t" .. k) end
   end
   if verbose then io.read() end
   return tab_entities
end

local input_labels = torch.Tensor()
local input_pubtator = torch.Tensor()


function test(networks, tagger, params, data, corpus)
   
   --initializing result tables (one per entity type)
   local tab_entities = {}
   for _, ent in pairs(tab_ent) do
      tab_entities[ent] = {}
      tab_entities[ent].ent_tp = 0
      tab_entities[ent].ent_fp = 0
      tab_entities[ent].ent_total = 0
   end

   local tab_entities2 = {}
   for _, ent in pairs(tab_ent) do
      tab_entities2[ent] = {}
      tab_entities2[ent].ent_tp = 0
      tab_entities2[ent].ent_fp = 0
      tab_entities2[ent].ent_fn = 0
   end

   local avg_f1 = torch.Tensor(#tab_ent)
   
   if params.dropout~=0 then
      networks.dropout.train=false
   end
   
   local pad = (params.wsz-1)/2
   
   --local findices = io.open(pathdata .. "/indices")

   --local output = io.open(params.rundir .. "/output_" .. corpus, "w")
   
   local verbose = false

   local cost = 0

   if params.brat then
      os.execute("rm prediction/* gold/*")
   end
   
   for idx=1, data.size do


      local entities_found = {}
      
      -- print("===========================================================================================================")
      -- print("===========================================================================================================")
      -- print("idx =============================================================" .. idx .. "=============================================================")
      -- print("===========================================================================================================")
      -- print("===========================================================================================================")
      local words = data.words[idx]
      --printw(words, data.wordhash)
      
      
      local labels, pubtator
      labels = input_labels:resize(words:size()):fill(data.labelhash["O"])--add padding
      for i=1,pad do
	 input_labels[i] = data.labelhash["PADDING"]
	 input_labels[ input_labels:size(1)-i+1 ] = data.labelhash["PADDING"]
      end
      pubtator = data.labels_pubtator[idx][1]
      

      if verbose then printw(words, data.wordhash) end
      local caps, tags, dict, pubtators, level1
      
      --printw(pubtators, data.pubtatorhash)
      --io.read()

      local tabwords = {}
      for w in data.words.sent[idx]:gmatch("[^ ]+") do
       	 table.insert(tabwords, w)
      end
      -- --print(tabwords)
      -- --print(data.words[idx])
      assert(#tabwords == data.words[idx]:size(1)-(2*pad))
      -- --printw(data.words[idx], data.wordhash)
      -- --io.read()
      -- --print(data.starts[idx]:reshape(1, data.starts[idx]:size(1)))
      -- --print(data.ends[idx]:reshape(1, data.ends[idx]:size(1)))

      
      local inputsize = words:size(1)-(params.wsz-1)

      local level = 1
      local onlyOthers = false
      local nb_ent = 0
      
      while (not onlyOthers) and level<5 do 
	 local input = {}
	 table.insert(input, words)
	 table.insert(input, labels)
	 if pfsz~=0 then
	    table.insert(input, pubtator)
	 end
	 -- print("\n===================================== INPUT TEST =====================================")
	 -- printw(input[1], data.wordhash)
	 -- printw(input[2], data.labelhash)
	 -- printw(input[3], data.pubtatorhash)
	 
	 local criterioninput = {}
	 criterioninput = networks:forward(input)
	 
	 score, path = tagger:forward_max(criterioninput)
	 --print("score " .. score)
	 
	 -- print("\n\nprediction test")
	 -- printw(path, data.labelhash)
	 -- io.read()
	 
	 -- print("path")
	 -- print(path)
	 -- print("gold")
	 -- print(data.chunks[idx])
	 --io.read()

	 --print("path")
	 --print(path)
	 lbl = path2lblbioes(path, params,data, data.labelhash)
	 -- print("lbl")
	 -- print(lbl)
	 -- exit()
	 local chunks = lbl2chunks(lbl)
	 --print("chunks")
	 --print(chunks)
	 for i=1,#chunks do
	    local first_word = chunks[i][3]
	    local first_indice = data.starts[idx][first_word] + first_word-1 -- + first_word-1 for spaces between words
	    local size = chunks[i][1]
	    local last_indice = data.ends[idx][first_word+size-1] + first_word-1+size -- + first_word-1+size for spaces between words
	    local _type = chunks[i][2]
	    local st = data.wordhash[ words[first_word+pad] ]
	    for i=first_word+1,first_word+size-1 do
	       st = st .. " " .. data.wordhash[ words[i+pad] ]
	    end
	    table.insert(entities_found, { {{first_indice, last_indice }}, _type, st })
	 end
	 
	 
	 if path:eq(data.chunkhash["O"]):sum()==path:size(1) then --only Others
	    onlyOthers = true
	 else
	    --print("old input")
	    --print(input)
	    -- print("old_labels")
	    -- printw(labels, data.labelhash)
	    data.clean_entities(entities_found)
	    --print("entities cleaned")
	    --print(entities_found)
	    data.getdag(entities_found)
	    --print(entities_found)
	    data.setlevel(entities_found)
	    --print(entities_found)
	    --printw(words, data.wordhash)
	    --print(data.starts[idx])
	    --print(data.ends[idx])
	    data._load_entity_indices(entities_found, data.starts[idx], data.ends[idx], "test")
	    --print(entities_found)
	    local newinput = data._extract_input_labels(entities_found, words, pad, data.labelhash)
	    --print("new_input")
	    --print(newinput[#newinput])
	    --printw(newinput[#newinput], data.labelhash)
	    
	    --labels:narrow(1, pad + 1, inputsize):copy(path) --no no and no
	    labels:copy(newinput[#newinput])
	    
	    -- print("new_labels")
	    --print(labels)
	    --printw(labels, data.labelhash)
	    --io.read()
	    pubtator = input_pubtator:resize(words:size()):fill(data.pubtatorhash["O"])
	    for i=1,pad do
	       input_labels[i] = data.labelhash["PADDING"]
	       input_labels[ input_labels:size(1)-i+1 ] = data.labelhash["PADDING"]
	    end
	    level = level + 1
	 end

	 if params.brat then
	    local tabstartend = gettabstartend(lbl, data, tabwords, idx)
	    local fann = io.open("prediction/" .. data.names[idx] .. ".ann", 'a')
	    writeinfile(tabstartend, fann,nb_ent)
	    fann:close()
	    nb_ent = nb_ent + #chunks
	 end
      end

      
      if params.brat then
	 --print(data.labels_pubtator[idx][1])
	 local pubtator_nopad = data.labels_pubtator[idx][1]:narrow(1,pad+1, data.labels_pubtator[idx][1]:size(1)-(2*pad)) 
	 --print(labels_nopad)
	 --print(pubtator_nopad)
	 lblpubtator = path2lblbioes(pubtator_nopad,params,data, data.pubtatorhash)
	 --gold level1 tags
	 --print(lblpubtator)
	 local tabstartend = gettabstartend(lblpubtator, data, tabwords, idx)
	 --print(tabstartend)
	 local fann = io.open("prediction/" .. data.names[idx] .. ".ann", 'a')
	 writeinfile(tabstartend, fann,nb_ent, "_p")
	 fann:close()
      end
      
      
      --convert gold entities (!!! only for contiguous entities !!!)
      local entities_gold = {}
      for i=1,#data.entities[idx] do
	 if #data.entities[idx][i][1]==1 then --only contiguous
	    table.insert(entities_gold, data.entities[idx][i])
	 end
      end

      -- print("entities_found")
      -- print(entities_found)
      -- print("\nentities_gold")
      -- print(entities_gold)
      -- io.read()
      
      --filling possible associations
      for i=1,#entities_found do
	 for j=1,#entities_gold do
	    if params.softmatch then
       	       if equal_ent(entities_found[i][2],entities_gold[j][2], params.hierarchy) and
	       match2(entities_found[i][1], entities_gold[j][1]) then
       		  if entities_found[i].asso then table.insert(entities_found[i].asso, j) else entities_found[i].asso = {j} end
       	       end	       
       	    else
       	       if equal_ent(entities_found[i][2],entities_gold[j][2], params.hierarchy) and
	       match(entities_found[i][1], entities_gold[j][1]) then
       		  if entities_found[i].asso then table.insert(entities_found[i].asso, j) else entities_found[i].asso = {j} end
       	       end
	       -- if entities_found[i][3]==entities_gold[j][3] and entities_found[i][1]==entities_gold[j][1] then --different tag
       	       -- end
       	    end
	 end
      end

      -- print("tab_ent")
      -- print(tab_ent)
      -- print("entities_found")
      -- print(entities_found)
      -- print("\nentities_gold") 
      -- print(entities_gold)
      -- io.read()
      
      --maximizing the number of match using the hopcroftKarp algorithm
      local g = graph:graph(#entities_found,#entities_gold)
      local total_etype = 0
      for i=1,#entities_found do
	 if entities_found[i].asso then
	    for j=1,#entities_found[i].asso do
	       g:addEdge(i, entities_found[i].asso[j])
	    end
	 end
      end
      local nmatch = g:hopcroftKarp() --true positive match
	 
      -- print("<<<<<=============================================================")
      -- print("entities_found")
      -- print(entities_found)
      -- print("entities_gold")
	 -- print(entities_gold)
      -- print("nmatch")
      -- print(nmatch)
      -- print(g)
      -- print("=============================================================>>>>>>")
      
      --adding validated associations in entities_found
      for i=1,#g.pairV do
	 local nmatch = 0
	 if g.pairV[i]~=0 then
	    --print("entity " .. i .. " ( " .. entities_gold[i][2] .. " / \"" .. entities_gold[i][4] .. "\" ) in gold")
	    --print("\t is associated with")
	    --print("entity " .. g.pairV[i] .. " ( " .. entities_found[ g.pairV[i] ][2] .. " / \"" .. entities_found[ g.pairV[i] ][3] .. "\" ) in pred\n") 
	    --if entities_gold[i][2]~=entities_found[ g.pairV[i] ][2] then print("diff"); io.read() end
	    entities_found[ g.pairV[i] ].match = entities_gold[i]
	 end
      end
      
      --computing tp and fp
      for i=1,#entities_found do
	 if entities_found[i].match then
	    tab_entities[ entities_found[i].match[2] ].ent_tp = tab_entities[ entities_found[i].match[2] ].ent_tp + 1
	    fill_tab_entities(params, data, tab_entities2, entities_found[i][2], entities_found[i].match[2])
	    entities_found[i].match.matched = true
	 else
	    tab_entities[ entities_found[i][2] ].ent_fp = tab_entities[ entities_found[i][2] ].ent_fp + 1 
	    fill_tab_entities(params, data, tab_entities2, entities_found[i][2], nil) --adding false positives
	 end
      end
      
      --computing target positive (tp + fn)
      for j=1,#entities_gold do
	 tab_entities[ entities_gold[j][2] ].ent_total = tab_entities[ entities_gold[j][2] ].ent_total + 1
	 if entities_gold.matched then
	    print("matched")
	 else
	    fill_tab_entities(params, data, tab_entities2, nil, entities_gold[j][2]) --adding false negatives
	 end
      end

      
      if params.brat then
      	 local fwords = io.open("gold/" .. data.names[idx] .. ".txt", "w")
      	 fwords:write(data.words.sent[idx])
      	 fwords:close()
      	 local fwords = io.open("prediction/" .. data.names[idx] .. ".txt", "w")
      	 fwords:write(data.words.sent[idx])
      	 fwords:close()

	 local counter_ent = 0
	 for i=1,#data.labels[idx] do
	    local labels_nopad = data.labels[idx][i]:narrow(1,pad+1, data.labels[idx][i]:size(1)-(2*pad)) 
	    --print(labels_nopad)
	    lblgold = path2lblbioes(labels_nopad,params,data, data.labelhash)
	    --gold level1 tags
	    local tabstartend = gettabstartend(lblgold, data, tabwords, idx)
	    --print(tabstartend)
	    local fann = io.open("gold/" .. data.names[idx] .. ".ann", 'a')
	    writeinfile(tabstartend, fann,counter_ent)
	    counter_ent = counter_ent + #tabstartend
	    fann:close()
	 end
      end
   end

   local tab_return = compute_prf1(tab_entities, tab_ent)
   local tab_return2 = compute_prf1_2(tab_entities2, tab_ent)
   print(tab_return)
   print(tab_return2)

   
   -- totals[totals:size(1)]=total_p
   
   -- local p = tp / (tp + fp)
   -- local r = tp / (total_p)
   -- local f1 = (2*p*r)/(p+r) 

   -- if params.verbose then
   --    print("total " .. total_p) 
   --    print("tp " .. tp)
   --    print("fp " .. fp)
   --    print("p " .. p)
   --    print("r " .. r)
   --    print("f " .. f1)
   -- end
   
   --output:close()
   
   
   -- local ps = tps:clone():add(fps:clone())
   -- ps = tps:clone():cdiv(ps)
   -- if params.verbose then
   --    print("precision")
   --    print(ps:clone():mul(100))
   --    print("recall")
   -- end
   -- local rs = tps:clone():cdiv(totals:clone())

   -- if params.verbose then
   --    print(rs:clone():mul(100))
   --    print("f1")
   -- end
   -- local f1s = ps:clone():cmul(rs:clone()):mul(2)
   -- f1s = f1s:cdiv( ps:clone():add(rs:clone()) )
   -- if params.verbose then
   --    print(f1s:mul(100))
   -- end
   
   -- for i=1,f1s:size(1) do
   --    print(string.format('%.2f',ps[i]) .. "\t" .. string.format('%.2f',rs[i]) .. "\t" .. string.format('%.2f',f1s[i]) .. "\t" ..data.chunkhash2[i])
   -- end
   
   -- print(f1s)
   -- exit()
   
   --local f1s = (2*
   
   --io.read()
   

   -- print("f1 : " .. (tonumber(f1) or 0))
   -- print(tonumber(precision) or 0)
   -- print(tonumber(recall) or 0)
   -- handle:close()
   -- print(line)
   
   if params.dropout~=0 then
      networks.dropout.train=true
   end
   
   return tab_return
end
