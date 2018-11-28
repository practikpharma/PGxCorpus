local inputwords = torch.Tensor()
local inputentities = torch.Tensor()

   
local tab_rel = {"isAssociatedWith", "isExplainedBy", "treats", "transports", "influences", "increases", "decreases", "causes", "metabolizes", "isEquivalentTo", "relation"}
local hierarchy_rel = {}
for i=1,#tab_rel do
   hierarchy_rel[tab_rel[i]] = {}
   hierarchy_rel[tab_rel[i]][tab_rel[i]] = true
end


hierarchy_rel["influences"]["causes"] = true
hierarchy_rel["influences"]["decreases"] = true
hierarchy_rel["influences"]["increases"] = true
hierarchy_rel["influences"]["metabolizes"] = true
hierarchy_rel["isAssociatedWith"]["isAssociatedWith"]=true
hierarchy_rel["isAssociatedWith"]["isExplainedBy"]=true
hierarchy_rel["isAssociatedWith"]["treats"]=true
hierarchy_rel["isAssociatedWith"]["transports"]=true
hierarchy_rel["isAssociatedWith"]["influences"]=true
hierarchy_rel["isAssociatedWith"]["increases"]=true
hierarchy_rel["isAssociatedWith"]["decreases"]=true
hierarchy_rel["isAssociatedWith"]["causes"]=true
hierarchy_rel["isAssociatedWith"]["metabolizes"]=true

function equal_rel(target, prediction, hierarchy)
   if hierarchy then
      return target==prediction or ((hierarchy_rel[prediction] and hierarchy_rel[prediction][target]) or (hierarchy_rel[target] and hierarchy_rel[target][prediction]))
   else
      return target==prediction
   end
end

function _forward(data, idx, ent1, ent2, network, criterion, params)
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
	       
   local output = network:forward(input)
   
   local target = data.relations:isrelated(idx, ent1, ent2)
   local cost = criterion:forward(output, target)

   return cost, output
end

function _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice)
   --if data.relationhash[class]=="int" then io.read() end
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
   
function test(network, data, params)
   
   local timer = torch.Timer()
   timer:reset()

   network:evaluate()
   
   local criterion = nn.ClassNLLCriterion()

   if params.dropout~=0 then
      if params.dp==1 or params.dp==3 or params.dp==4 then
	 network.dropout:evaluate()
      end
   end

   if params.brat then
      os.execute("rm prediction/* gold/*")
   end
   
   local cost = 0
   local nforward = 0
   local toto = {0,0,0,0,0}
   
   local confusion_matrix = torch.Tensor(#data.relationhash, #data.relationhash):fill(0)
   
   local precision_recall = {}
   for i=1,#data.relationhash do
      precision_recall[i] = {totalpos=0, truepos=0, falsepos=0}
   end
   
   for idx=1,data.size do
      --print(idx .. " " .. data.words[idx]:size(1) .. "/" .. data.size)
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
	       if data.relations:isrelated(idx, ent1, ent2)~=data.relationhash.null then
		  print(data.entities[idx][ent1][3])
		  print(data.entities[idx][ent2][3])
		  print(ent1, ent2)
		  printw(data.words[idx], data.wordhash)
		  error("")
	       end
	       --These entities are nested or overlapp and thus are not related
	    else
	       --print("relation between " .. ent1 .. " and " .. ent2 .. " (" .. data.relations:isrelated(idx, ent1, ent2) .. ")")
	       local c, output = _forward(data, idx, ent1, ent2, network, criterion, params) 
	       cost = cost + c
	       
	       local max_1, indice_1 = output:max(1)
	       indice_1 = indice_1[1]
	       max_1 = max_1[1]
	       
	       if params.oriented then
		  local c, output_2 = _forward(data, idx, ent2, ent1, network, criterion, params) 
		  cost = cost + c
		  local max_2, indice_2 = output_2:max(1)
		  indice_2 = indice_2[1]
		  max_2 = max_2[1]


		  if true then --new version
		     if params.hierarchy then
			local class = data.relations:isrelated(idx, ent1, ent2)
			if class==data.relationhash["isAssociatedWith"] then --gold is isAssociatedWith (the only undirected relation)
			   if indice_1==data.relationhash["isAssociatedWith"] and indice_2==data.relationhash["isAssociatedWith"] then
			      --since isAssociatedWith is undirected, only one true positive isAssociatedWith is counted
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			      --print("toto1")
			      toto[1] = toto[1]+1
			   elseif indice_1~=data.relationhash["null"] and indice_2~=data.relationhash["null"] then
			      --relation in both direction. Since isAssociatedWith is undirected, both can be correct 
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_2)
			   elseif indice_1~=data.relationhash["null"] then --only one relation between ent1 and ent2 
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			      --we do not penalize the model for not finding the isAssociated in the other direction
			      --since it is undirected
			      --print("toto2")
			      toto[2] = toto[2]+1
			   elseif indice_2~=data.relationhash["null"] then --only one relation between ent2 and ent1 
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_2)
			      --print("toto3")
			      toto[3] = toto[3]+1
			      --same comment as the one above
			   else
			      local class = data.relations:isrelated(idx, ent1, ent2)
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			      local class = data.relations:isrelated(idx, ent2, ent1)
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_2)
			   end
			else --gold is not isAssociatedWith
			   if indice_1==data.relationhash["isAssociatedWith"] and indice_2==data.relationhash["isAssociatedWith"] then
			      --since isAssociatedWith is undirected, only one isAssociatedWith is considered
			      local class1 = data.relations:isrelated(idx, ent1, ent2)
			      local class2 = data.relations:isrelated(idx, ent2, ent1)
			      local class = class1~=data.relationhash["null"] and class1 or class2
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			      --print("toto5")
			      --toto[5] = toto[5]+1
			   elseif indice_1~=data.relationhash["null"] and indice_2~=data.relationhash["null"] then
			      --relation in both direction. Let's choose the best scoring one (the other one is set to "null").
			      if max_1>max_2 then indice_2 = data.relationhash["null"]
			      else indice_1 = data.relationhash["null"] end
			      local class = data.relations:isrelated(idx, ent1, ent2)
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			      local class = data.relations:isrelated(idx, ent2, ent1)
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_2)
			   else
			      local class = data.relations:isrelated(idx, ent1, ent2)
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			      local class = data.relations:isrelated(idx, ent2, ent1)
			      _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_2)
			   end
			end
		     else --no hierarchy
			error("to do")
		     end
		  
		  elseif true then --all relation in both direction
		     local class = data.relations:isrelated(idx, ent1, ent2)
		     _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
		     local class = data.relations:isrelated(idx, ent2, ent1)
		     _confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_2)
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
		  else
		     if indice_1==data.relationhash["isAssociatedWith"] and indice_2==data.relationhash["isAssociatedWith"] then
			print("toto")
			local class = data.relations:isrelated(idx, ent1, ent2)
			_confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			if params.brat then
			   if indice_1~=data.relationhash["null"] then
			      if not relations_predicted[ent1] then
				 relations_predicted[ent1] = {}
			      end
			      relations_predicted[ent1][ent2]=indice_1
			   end
			end
			local class = data.relations:isrelated(idx, ent1, ent2)
			_confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			if params.brat then
			   if indice_1~=data.relationhash["null"] then
			      if not relations_predicted[ent1] then
				 relations_predicted[ent1] = {}
			      end
			      relations_predicted[ent1][ent2]=indice_1
			   end
			end
		     elseif indice_1~=data.relationhash["null"] and indice_2~=data.relationhash["null"] then
			--relation in both direction. Let's choose the best scoring one (the other one is set to "null").
			--print(max_1 .. " " .. max_2)
			if max_1>max_2 then
			   indice_2 = data.relationhash["null"]
			else
			   indice_1 = data.relationhash["null"]
			end
			local class = data.relations:isrelated(idx, ent1, ent2)
			_confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			local class = data.relations:isrelated(idx, ent2, ent1)
			_confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_2)
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
		     else
			local class = data.relations:isrelated(idx, ent1, ent2)
			_confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_1)
			local class = data.relations:isrelated(idx, ent2, ent1)
			_confusion_matrix(data, params, precision_recall, confusion_matrix, class, indice_2)
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

		  -- if data.relationhash[indice]=="isEquivalentTo" or data.relationhash[class]=="isEquivalentTo"then
		  --    print("expected class " .. data.relationhash[class] .. " | predicted class " .. data.relationhash[indice] .. " (" .. ent1 .. " " .. ent2 .. ")")
		  -- end
		  
	       else --not oriented
		  local class = data.relations:isrelated(idx, ent1, ent2)
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
	 local fwords = io.open("gold/" .. data.names[idx] .. ".txt", "w")
	 
	 fwords:write(data.words.sent[idx])
	 fwords:close()
	 local fwords = io.open("prediction/" .. data.names[idx] .. ".txt", "w")
	 fwords:write(data.words.sent[idx])
	 fwords:close()

	 --gold entities in gold file
	 local fann = io.open("gold/" .. data.names[idx] .. ".ann", "w")
	 for i=1,#data.entities[idx] do
	    fann:write(data.entities[idx][i][3] .. "\t" ..data.entities[idx][i][2] .. " ")
	    fann:write(data.entities[idx][i][1][1][1] .. " " .. data.entities[idx][i][1][1][2])
	    for j=2,#data.entities[idx][i][1] do
	       fann:write(";" .. data.entities[idx][i][1][j][1] .. " " .. data.entities[idx][i][1][j][2])
	    end
	    fann:write("\t" .. data.entities[idx][i][4] .. "\n")
	 end
	 
	 --gold relation in gold
	 local r = 0
	 --print(data.relations[idx])
	 for k1,v1 in pairs(data.relations[idx]) do
	    for k2, v2 in pairs(v1) do
	       r = r + 1
	       --print(k1 .. " " .. k2 .. " : " .. v2)
	       local brat_ent1 = data.entities[idx][k1][3]
	       local brat_ent2 = data.entities[idx][k2][3]
	       fann:write("R" .. r .. "\t" .. data.relationhash[v2] .. " Arg1:" .. brat_ent1 .. " Arg2:" .. brat_ent2 .. "\n")
	    end
	 end
	 fann:close()

	 
	 --gold entities in pred file
	 local fann = io.open("prediction/" .. data.names[idx] .. ".ann", "w")
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

   print("nb toto " .. toto[1] .. " + " .. toto[2] .. " + " .. toto[3] .. " + " .. toto[4] .. " + " .. toto[5] .. " = " .. toto[1]+toto[2]+toto[3]+toto[4]+toto[5] )

   
   print(data.relationhash)
   
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
   
   local t = timer:time().real
   print(string.format('test corpus processed in %.2f seconds (%.2f sentences/s)', t, data.size/t))

   
   cost = cost/nforward

   local class_to_consider = {}
   if params.onerelation then
      class_to_consider = {2}
   else
      for k,v in pairs(params.onlylabel) do
	 table.insert(class_to_consider, data.relationhash[k])
      end
   end
   
   --computing evaluation measures
   --macro-average (avg r and p over all categories)
   local recalls, precisions = {}, {}
   local macro_R, macro_P = 0, 0
   local nb_recall, nb_precision = 0,0
   for k,i in pairs(class_to_consider) do
      --print(data.relationhash[i])
      local a = precision_recall[i].truepos
      local b = precision_recall[i].totalpos

      --recalls[i] = (a==0 and b==0 and 0 or a/b)
      --recalls[i] = a/b
      recalls[i] = (b==0 and 1 or a/b)

      --print("a " .. a .. " b " .. b .. " R " .. recalls[i])
      local a = precision_recall[i].truepos
      local b = precision_recall[i].truepos + precision_recall[i].falsepos

      --precisions[i] = (a==0 and b==0 and 0 or a/b)
      --precisions[i] = a/b
      precisions[i] = (b==0 and 1 or a/b)

      --print("a " .. a .. " b " .. b .. " P " .. precisions[i] .. " fp " .. precision_recall[i].falsepos)

      -- if recalls[i]==recalls[i] then
      -- 	 macro_R = macro_R + recalls[i] 
      -- 	 nb_recall = nb_recall + 1
      -- else
      -- 	 --macro_R = macro_R + 0; nb_recall = nb_recall + 1
      -- end
      -- if precisions[i]==precisions[i] then
      -- 	 macro_P = macro_P + precisions[i] 
      -- 	 nb_precision = nb_precision + 1
      -- else
      -- 	 --macro_P = macro_P + 0; nb_precision = nb_precision + 1
      -- end
      macro_R = macro_R + ((recalls[i]==recalls[i]) and recalls[i] or 0)
      macro_P = macro_P + ((precisions[i]==precisions[i]) and precisions[i] or 0)
   end
   --macro_R = macro_R/nb_recall
   --macro_P = macro_P/nb_precision
   macro_R = macro_R / (#class_to_consider)
   macro_P = macro_P / (#class_to_consider)
   local macro_f1score = (2 * macro_R * macro_P) / (macro_R + macro_P)
   macro_f1score = macro_f1score==macro_f1score and macro_f1score or 0
   
   --micro average precision (sum truepos, falsepos, totalpos and compute P and R)
   local _truepos, _falsepos, _totalpos = 0, 0, 0
   for k,i in pairs(class_to_consider) do
      _truepos = _truepos + precision_recall[i].truepos
      _totalpos = _totalpos + precision_recall[i].totalpos
      _falsepos = _falsepos + precision_recall[i].falsepos
   end
   local a = _truepos
   local b = _totalpos
   --print("a " .. a .. " b " .. b)
   --local micro_R = (a==0 and b==0 and 0 or a/b)
   --local micro_R = a/b
   local micro_R = (b==0 and 1 or a/b) 
   
   local a = _truepos
   local b = _truepos + _falsepos
   --print("a " .. a .. " b " .. b)
   --local micro_P = (a==0 and b==0 and 0 or a/b)
   --local micro_P = a/b
   local micro_P = (b==0 and 1 or a/b)
   
   local micro_f1score = (2 * micro_R * micro_P) / (micro_R + micro_P)
   micro_f1score = micro_f1score==micro_f1score and micro_f1score or 0
   
   --print(data.relationhash)
   print("\t\tP\tR")
   table.sort(class_to_consider, function(a,b) return a<b end) 
   print(class_to_consider)
   --for k,i in pairs(class_to_consider) do
   for j=1,#class_to_consider do
      local i = class_to_consider[j]
      print("Class " .. i .. ":\t" .. string.format('%.2f',precisions[i]) .. "\t" .. string.format('%.2f',recalls[i]) .. " " .. data.relationhash[i])
   end
   
   local tab_return = {}
   for k,i in pairs(class_to_consider) do
      --print(data.relationhash[i])
      tab_return[ data.relationhash[i] ] = {precision = precisions[i], recall = recalls[i], f1 = (2*precisions[i]*recalls[i])/(precisions[i]+recalls[i]) }
      tab_return[ data.relationhash[i] ].precision = tab_return[ data.relationhash[i] ].precision==tab_return[ data.relationhash[i] ].precision and tab_return[ data.relationhash[i] ].precision or 0
      tab_return[ data.relationhash[i] ].recall = tab_return[ data.relationhash[i] ].recall==tab_return[ data.relationhash[i] ].recall and tab_return[ data.relationhash[i] ].recall or 0
      tab_return[ data.relationhash[i] ].f1 = tab_return[ data.relationhash[i] ].f1==tab_return[ data.relationhash[i] ].f1 and tab_return[ data.relationhash[i] ].f1 or 0
   end
   tab_return["macro_avg"] = {precision=macro_P or 0, recall=macro_R or 0, f1=macro_f1score or 0}
   tab_return["micro_avg"] = {precision=micro_P or 0, recall=micro_R or 0, f1=micro_f1score or 0}
   tab_return.cost = cost
   
   print(tab_return["influences"])
   print(precision_recall[data.relationhash["influences"]])
   
   --return (macro_P or 0), (macro_R or 0), (macro_f1score or 0), cost, micro_P, micro_R, micro_f1score
   return tab_return
end
