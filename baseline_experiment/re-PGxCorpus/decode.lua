local inputwords = torch.Tensor()
local inputentities = torch.Tensor()
   
local tab_rel = {"isAssociatedWith", "influences", "causes", "increases", "decreases", "treats", "isEquivalentTo"}
local hierarchy_rel = {}
for i=1,#tab_rel do
   hierarchy_rel[tab_rel[i]] = {}
   hierarchy_rel[tab_rel[i]][tab_rel[i]] = true
end
hierarchy_rel["influences"]["influences"] = true
hierarchy_rel["influences"]["causes"] = true
hierarchy_rel["influences"]["decreases"] = true
hierarchy_rel["influences"]["increases"] = true
hierarchy_rel["isAssociatedWith"]["isAssociatedWith"]=true
hierarchy_rel["isAssociatedWith"]["treats"]=true
hierarchy_rel["isAssociatedWith"]["influences"]=true
hierarchy_rel["isAssociatedWith"]["increases"]=true
hierarchy_rel["isAssociatedWith"]["decreases"]=true
hierarchy_rel["isAssociatedWith"]["causes"]=true


local back_hierarchy_rel = {}
back_hierarchy_rel["treats"] = "isAssociatedWith"
back_hierarchy_rel["influences"] = "isAssociatedWith"
back_hierarchy_rel["decreases"] = "influences"
back_hierarchy_rel["increases"] = "influences"
back_hierarchy_rel["causes"] = "influences"
back_hierarchy_rel["isAssociatedWith"] = nil
back_hierarchy_rel["isEquivalentTo"] = nil


function equal_rel(target, prediction, hierarchy)
   if hierarchy then
      return target==prediction or ((hierarchy_rel[prediction] and hierarchy_rel[prediction][target]) or (hierarchy_rel[target] and hierarchy_rel[target][prediction]))
   else
      return target==prediction
   end
end


function _forward(data, idx, ent1, ent2, networks, criterion, params)
   local words = data.words[idx]
   local entities = data.entities.getent(data, idx, ent1, ent2)
   
   local input = {words}
   if params.tfsz~=0 then table.insert(input, data.entities.getenttags(data, idx, ent1, ent2)) end
   if params.pfsz~=0 then table.insert(input, data.pos[idx]) end
   if params.rdfsz~=0 then
      table.insert(input, data.get_relative_distance(entities, 1))
      table.insert(input, data.get_relative_distance(entities, 2))
   end
   if params.nestenttype>0 then
      local nests = data.entities.getnestenttype(data, idx, ent1, ent2)
      for i=1,#nests do
	 table.insert(input, nests[i])
      end
   end
   table.insert(input, entities)
   
   if params.anonymize then
      input = anonymize(words, data.entities[idx], ent1, ent2, data, params)
   end
   
   local output
   if params.arch=="mccnn" then
      output = networks[1]:forward(input)
      for n=2,#networks do
	 output:add(networks[n]:forward(input))
      end
   elseif params.arch=="treelstm" then
      error("to do")
      local t =  data.trees.gettrees(data, idx, ent1, ent2)
      output = network:forward(t, input)
      network.treelstm:clean(t)
   else
      error("")
   end
   
   local target = data.relations:isrelated(idx, ent1, ent2)
   
   return output
end

function _confusion_matrix2(data, params, confusion_matrix, target, prediction, verbose)
   if false and target=="increases" then verbose=true else verbose = false end
   
   if verbose then
      print("==================================")
      print("target " .. target)
      print("prediction " .. prediction)
   end
   if verbose then print("confusion_matrix before"); print_confusion_matrix(data, confusion_matrix); end 

   if params.hierarchy then
      local target_indice = data.relationhash[target]
      local prediction_indice = data.relationhash[prediction]
      
      if target~="null" and hierarchy_rel[target][prediction] then
	 if verbose then print("the prediction (" .. prediction .. ") is more specific than the target (" .. target .. ")") end
	 local current = prediction
	 while current~=target do
	    if verbose then print(current .. " is a false positive 1") end
	    local current_indice = data.relationhash[current]
	    confusion_matrix[ data.relationhash["null"] ][current_indice] = confusion_matrix[ data.relationhash["null"] ][current_indice] + 1 
	    current = back_hierarchy_rel[current]
	 end
	 while current do
	    if verbose then print(current .. " is a true positive 2") end
	    local current_indice = data.relationhash[current]
	    confusion_matrix[current_indice][current_indice] = confusion_matrix[current_indice][current_indice] + 1 
	    current = back_hierarchy_rel[current]
	 end
      elseif prediction~="null" and hierarchy_rel[prediction][target] then
	 if verbose then print("the prediction (" .. prediction .. ") is less specific than the target (" .. target .. ")") end
	 local current = target
	 while current~=prediction do
	    if verbose then print(current .. " is a false negative 3") end
	    local current_indice = data.relationhash[current]
	    confusion_matrix[current_indice][ data.relationhash["null"] ] = confusion_matrix[current_indice][ data.relationhash["null"] ] + 1 
	    current = back_hierarchy_rel[current]
	 end
	 while current do
	    if verbose then print(current .. " is a true positive 4") end
	    local current_indice = data.relationhash[current]
	    confusion_matrix[current_indice][current_indice] = confusion_matrix[current_indice][current_indice] + 1 
	    current = back_hierarchy_rel[current]
	 end
      else
	 local target_ancestors = {}
	 local current = back_hierarchy_rel[target]
	 while current do
	    target_ancestors[current]=true
	    current = back_hierarchy_rel[current]
	 end
	 local current = prediction
	 while current do
	    if target_ancestors[current] then
	       if verbose then print(current .. " is a true positive 5") end
	       local current_indice = data.relationhash[current]
	       confusion_matrix[current_indice][current_indice] = confusion_matrix[current_indice][current_indice] + 1
	       target_ancestors[current] = false
	    else
	       if verbose then print(current .. " is a false positive and " .. target .. " is a false negative 6") end
	       local current_indice = data.relationhash[current]
	       confusion_matrix[ target_indice ][ current_indice ] = confusion_matrix[ target_indice ][current_indice] + 1
	    end
	    current = back_hierarchy_rel[current]
	 end

	 for k,v in pairs(target_ancestors) do
	    if v then --false negative
	       local k_indice = data.relationhash[k]
	       if verbose then print(k .. " is a false negative 7") end
		  confusion_matrix[k_indice][ data.relationhash["null"] ] = confusion_matrix[k_indice][ data.relationhash["null"] ] + 1 
	    end
	 end
      end
   else
      if target==prediction then
	 if verbose then print(target .. " is a true positive 5") end
	 confusion_matrix[data.relationhash[target]][data.relationhash[target]] = confusion_matrix[data.relationhash[target]][data.relationhash[target]] + 1
      else
	 -- print("error")
	 if false then
	    printw(words, data.wordhash)
	    print(data.relationhash[data.relationhash[target]] .. " but classified as " .. data.relationhash[data.relationhash[prediction]]) 
	    io.read()
	 end
	 if verbose then print(prediction .. " is a false positive 5") end
	 if verbose then print(target .. " is a false negative 5") end
	 confusion_matrix[data.relationhash[target]][data.relationhash[prediction]] = confusion_matrix[data.relationhash[target]][data.relationhash[prediction]] + 1
      end
   end
   if verbose then print("confusion_matrix after"); print_confusion_matrix(data, confusion_matrix); io.read() end 
   
end

-- _confusion_matrix2(nil, {hierarchy=true}, torch.Tensor({10,10}), "decreases", "influences")
-- _confusion_matrix2(nil, {hierarchy=true}, torch.Tensor({10,10}), "influences", "decreases")
-- _confusion_matrix2(nil, {hierarchy=true}, torch.Tensor({10,10}), "decreases", "increases")
-- _confusion_matrix2(nil, {hierarchy=true}, torch.Tensor({10,10}), "isEquivalentTo", "increases")
-- _confusion_matrix2(nil, {hierarchy=true}, torch.Tensor({10,10}),  "increases", "isEquivalentTo")
-- exit()

function _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice)
   --if data.relationhash[class]=="int" then io.read() end
   --print(class)
   precision_recall[class].totalpos = precision_recall[class].totalpos +1
   if equal_rel(data.relationhash[class], data.relationhash[indice], params.hierarchy) then
      precision_recall[class].truepos = precision_recall[class].truepos+1
      confusion_matrix[class][class] = confusion_matrix[class][class] + 1
   else
      -- print("error")
      if false then
	 printw(words, data.wordhash)
	 print(data.relationhash[class] .. " but classified as " .. data.relationhash[indice]) 
	 io.read()
      end
      precision_recall[indice].falsepos = precision_recall[indice].falsepos + 1
      confusion_matrix[class][indice] = confusion_matrix[class][indice] + 1
   end
end

function print_confusion_matrix(data, confusion_matrix)
   for i=1,#data.relationhash do
      for j=1,confusion_matrix[i]:size(1) do
	 io.write(confusion_matrix[i][j] .. "\t")
      end
      io.write(data.relationhash[i] .. " " .. confusion_matrix[i]:sum())
      io.write("\n")
   end
   --print(confusion_matrix)

   for i=2,#data.relationhash do
      local r = confusion_matrix[i]:sum()==0 and 1 or confusion_matrix[i][i]/confusion_matrix[i]:sum()
      local p = confusion_matrix:narrow(2,i,1):sum()==0 and 1 or confusion_matrix[i][i]/ confusion_matrix:narrow(2,i,1):sum()
      print(data.relationhash[i] .. " p " .. p .. " r " .. r)
   end
end

function compute_micro_macro(data, confusion_matrix, class_to_consider)
   local recalls, precisions = {}, {}
   local macro_R, macro_P = 0, 0
   for k,i in pairs(class_to_consider) do
      recalls[i] = confusion_matrix[i]:sum()==0 and 1 or confusion_matrix[i][i]/confusion_matrix[i]:sum()
      precisions[i] = confusion_matrix:narrow(2,i,1):sum()==0 and 1 or confusion_matrix[i][i]/ confusion_matrix:narrow(2,i,1):sum()
      
      macro_R = macro_R + recalls[i]
      macro_P = macro_P + precisions[i]
   end
   macro_R = macro_R / (#class_to_consider)
   macro_P = macro_P / (#class_to_consider)

   local macro_f1score = (2 * macro_R * macro_P) / (macro_R + macro_P)
   macro_f1score = macro_f1score==macro_f1score and macro_f1score or 0



   local true_positives, total_positives, false_positives = 0,0,0
   for k,i in pairs(class_to_consider) do
      true_positives = true_positives + confusion_matrix[i][i]
      total_positives = total_positives + confusion_matrix[i]:sum()
      false_positives = false_positives + confusion_matrix:narrow(2,i,1):sum() - confusion_matrix[i][i]
   end
   micro_R = true_positives / total_positives
   micro_P = true_positives / (true_positives + false_positives) 
   micro_f1score = (2 * micro_R * micro_P) / (micro_R + micro_P)
   

   
   local tab_return = {}
   for k,i in pairs(class_to_consider) do
      --print(data.relationhash[i])
      tab_return[ data.relationhash[i] ] = {precision = precisions[i], recall = recalls[i], f1 = (2*precisions[i]*recalls[i])/(precisions[i]+recalls[i]) }
      tab_return[ data.relationhash[i] ].precision = tab_return[ data.relationhash[i] ].precision==tab_return[ data.relationhash[i] ].precision and tab_return[ data.relationhash[i] ].precision or 0
      tab_return[ data.relationhash[i] ].recall = tab_return[ data.relationhash[i] ].recall==tab_return[ data.relationhash[i] ].recall and tab_return[ data.relationhash[i] ].recall or 0
      tab_return[ data.relationhash[i] ].f1 = tab_return[ data.relationhash[i] ].f1==tab_return[ data.relationhash[i] ].f1 and tab_return[ data.relationhash[i] ].f1 or 0
   end
   tab_return["macro_avg"] = {precision=macro_P or 0, recall=macro_R or 0, f1=macro_f1score}
   tab_return["micro_avg"] = {precision=micro_P or 0, recall=micro_R or 0, f1=micro_f1score}
   
   return tab_return
end

function decode(networks, data, params)

   local timer = torch.Timer()
   timer:reset()
   
   for i=1,#networks do
      networks[i]:evaluate()
   end

   if params.dropout~=0 then
      if params.dp==1 or params.dp==3 or params.dp==4 then
	 for i=1,#networks do
	    networks[i].dropout:evaluate()
	 end
      end
   end

   if params.brat~='' then
      os.execute("rm " .. params.brat .. "/*")
   end
   
   local nforward = 0
   local toto = {0,0,0,0,0}
   
   local confusion_matrix = torch.Tensor(#data.relationhash, #data.relationhash):fill(0)
   local confusion_matrix2 = torch.Tensor(#data.relationhash, #data.relationhash):fill(0)
   
   local precision_recall = {}
   for i=1,#data.relationhash do
      precision_recall[i] = {totalpos=0, truepos=0, falsepos=0}
   end
   
   for idx=1,data.size do
      print(idx .. " " .. data.words[idx]:size(1) .. "/" .. data.size .. " " .. data.names[idx])
      -- print(data.words[idx]:size(1))
      --printw(data.words[idx], data.wordhash)
      
      if data.entities.nent(data,idx)<2 then
      else
	 local n = data.entities.nent(data,idx)
	 nforward = nforward + ((n * (n-1))/2)
      end      

      local relations_predicted = {}

      for ent1=1,data.entities.nent(data,idx) do
	 for ent2=ent1+1,data.entities.nent(data,idx) do --
	    if is_included(data.entities[idx][ent1][1], data.entities[idx][ent2][1]) or is_included(data.entities[idx][ent2][1], data.entities[idx][ent1][1]) or overlapp(data.entities[idx][ent1][5], data.entities[idx][ent2][5]) then
	    else
	       --print("test " .. idx .. " relation between " .. ent1 .. " and " .. ent2 .. " (" .. data.relations:isrelated(idx, ent1, ent2) .. ")")
	       local output = _forward(data, idx, ent1, ent2, networks, criterion, params) 
	       
	       local max_1, indice_1 = output:max(1)
	       indice_1 = indice_1[1]
	       max_1 = max_1[1]
	       
	       if params.oriented then
		  local output_2 = _forward(data, idx, ent2, ent1, networks, criterion, params) 
		  local max_2, indice_2 = output_2:max(1)
		  indice_2 = indice_2[1]
		  max_2 = max_2[1]
		  
		  if true then --new version
		     if true or params.hierarchy then
			local class = data.relations:isrelated(idx, ent1, ent2, true)
			if class==data.relationhash["isAssociatedWith"] then --gold is isAssociatedWith (the only undirected relation)
			   
			else --gold is not isAssociatedWith
			   if indice_1~=data.relationhash["null"] and indice_2~=data.relationhash["null"] then
			      --relation in both direction. Let's choose the best scoring one (the other one is set to "null").
			      if max_1>max_2 then
				 indice_2 = data.relationhash["null"]
				 if params.brat then
				    if not relations_predicted[ent1] then
				       relations_predicted[ent1] = {}
				    end
				    relations_predicted[ent1][ent2]=indice_1
				 end
			      else
				 indice_1 = data.relationhash["null"]
				 if params.brat then
				    if not relations_predicted[ent2] then
				       relations_predicted[ent2] = {}
				    end
				    relations_predicted[ent2][ent1]=indice_2
				 end
			      end
			      
			   else
			      if params.brat then
				 if indice_1~=data.relationhash["null"] then
				    if not relations_predicted[ent1] then
				       relations_predicted[ent1] = {}
				    end
				    relations_predicted[ent1][ent2]=indice_1
				 end
			      end
			      if params.brat then
				 if indice_2~=data.relationhash["null"] then
				    if not relations_predicted[ent2] then
				       relations_predicted[ent2] = {}
				    end
				    relations_predicted[ent2][ent1]=indice_2
				 end
			      end
			   end
			end
		     else --no hierarchy
			error("???")
		     end
		  
		  elseif true then --all relation in both direction
		  else
		  end
		  
	       else --not oriented
		  local class = data.relations:isrelated(idx, ent1, ent2, true)
		  _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
		  if params.brat then
		     if indice_1~=data.relationhash["null"] then
			if not relations_predicted[ent1] then
			   relations_predicted[ent1] = {}
			end
			relations_predicted[ent1][ent2]=indice_1
		     end
		  end

	       end
	       
	       
	       --print("expected class " .. data.relationhash[class] .. " | predicted class " .. data.relationhash[indice] .. " (" .. ent1 .. " " .. ent2 .. ")" )
	       
	    end
	 end
      end
      
      if params.brat then
	 local fwords = io.open(params.brat .. "/" .. data.names[idx] .. ".txt", "w")
	 fwords:write(data.words.sent[idx])
	 fwords:close()

	 --gold entities in pred file
	 local fann = io.open(params.brat .. "/" .. data.names[idx] .. ".ann", "w")
	 for i=1,#data.entities[idx] do
	    fann:write(data.entities[idx][i][3] .. "\t" ..data.entities[idx][i][2] .. " ")
	    fann:write(data.entities[idx][i][1][1][1] .. " " .. data.entities[idx][i][1][1][2])
	    for j=2,#data.entities[idx][i][1] do
	       fann:write(";" .. data.entities[idx][i][1][j][1] .. " " .. data.entities[idx][i][1][j][2])
	    end
	    fann:write("\t" .. data.entities[idx][i][4] .. "\n")
	 end
	 
	 --pred relation in pred file
	 --print(relations_predicted)
	 local r = 0
	 --print(relations_predicted)
	 for k1,v1 in pairs(relations_predicted) do
	    for k2, v2 in pairs(v1) do
	       local brat_ent1 = data.entities[idx][k1][3]
	       local brat_ent2 = data.entities[idx][k2][3]
	       r = r + 1
	       --print(k1 .. " " .. k2 .. " : " .. v2)
	       fann:write("R" .. r .. "\t" .. data.relationhash[v2] .. " Arg1:" .. brat_ent1 .. " Arg2:" .. brat_ent2 .. "\n")
	    end
	 end
	 fann:close()
	 --io.read()

      end

      --io.read()
      
   end
  
   local t = timer:time().real
   print(string.format('Corpus decoded in %.2f seconds (%.2f sentences/s)', t, data.size/t))
        
end
