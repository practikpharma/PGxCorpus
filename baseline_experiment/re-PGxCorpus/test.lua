local inputwords = torch.Tensor()
local inputentities = torch.Tensor()

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

   local confusion_matrix = torch.Tensor(#data.relationhash, #data.relationhash):fill(0)
   
   local precision_recall = {}
   for i=1,#data.relationhash do
      precision_recall[i] = {totalpos=0, truepos=0, falsepos=0}
   end
   
   for idx=1,data.size do

      --print(idx .. " " .. data.words[idx]:size(1) .. "/" .. data.size)
      -- print(data.words[idx]:size(1))
      -- print(data.trees[idx])
      --print(data.ids[idx])
      --printw(data.words[idx], data.wordhash)

      -- print(data.names[idx])
      -- print(data.relations[idx])
      -- print(data.entities[idx])
      
      
      local words = data.words[idx]
      if (params.dp==2 or params.dp==3  or params.rnn=="lstm" or params.rnn=="cnn") then words = words:view(1,words:size(1)) end
      
      if data.entities.nent(data,idx)<2 then
      else
	 local n = data.entities.nent(data,idx)
	 nforward = nforward + ((n * (n-1))/2)
      end      

      local relations_predicted = {}
      
      for ent1=1,data.entities.nent(data,idx) do
	 for ent2=ent1+1,data.entities.nent(data,idx) do
	    if is_included(data.entities[idx][ent1][1], data.entities[idx][ent2][1]) or is_included(data.entities[idx][ent2][1], data.entities[idx][ent1][1]) then
	       --These entities are nested and thus not related
	    else
	       --print("relation between " .. ent1 .. " and " .. ent2 .. " (" .. data.relations:isrelated(idx, ent1, ent2) .. ")")
	       local entities = data.entities.getent(data, idx, ent1, ent2)
	       if (params.dp==2 or params.dp==3 or params.rnn=="lstm" or params.rnn=="cnn") then entities = entities:view(1, entities:size(1)) end
	       
	       local input = {words}
	       if params.tfsz~=0 then table.insert(input, data.entities.getenttags(data, idx, ent1, ent2)) end
	       if params.pfsz~=0 then table.insert(input, data.pos[idx]) end
	       if params.rdfsz~=0 then
		  table.insert(input, data.get_relative_distance(entities, 1))
		  table.insert(input, data.get_relative_distance(entities, 2))
	       end
	       table.insert(input, entities)
	       
	       local output
	       output = network:forward(input)
	       
	       local target = data.relations:isrelated(idx, ent1, ent2)

	       cost = cost + criterion:forward(output, target)
	       
	       if (params.dp==2 or params.dp==3 or params.rnn=="lstm" or params.rnn=="cnn") then output = output[1] end

	       local max, indice = output:max(1)
	       indice = indice[1]

	       --print(target .. " " .. indice)
	       
	       local class = data.relations:isrelated(idx, ent1, ent2)
	       --if data.relationhash[class]=="int" then io.read() end
	       precision_recall[class].totalpos = precision_recall[class].totalpos +1
	       if class==indice then
		  precision_recall[indice].truepos = precision_recall[indice].truepos+1
	       else
		  -- print("error")
		  if false then
		     printw(words, data.wordhash)
		     print(data.relationhash[class] .. " but classified as " .. data.relationhash[indice]) 
		     io.read()
		  end
		  precision_recall[indice].falsepos = precision_recall[indice].falsepos + 1
	       end
	       confusion_matrix[class][indice] = confusion_matrix[class][indice] + 1

	       if params.brat then
		  if indice~=data.relationhash["null"] then
		     if not relations_predicted[ent1] then
			relations_predicted[ent1] = {}
		     end
		     relations_predicted[ent1][ent2]=indice
		  end
	       end
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
   --macro-average (avg min and presicion over all categories)
   local recalls, precisions = {}, {}
   local macro_R, macro_P = 0, 0
   for k,i in pairs(class_to_consider) do
      --print(data.relationhash[i])
      local a = precision_recall[i].truepos
      local b = precision_recall[i].totalpos
      recalls[i] = (a==0 and b==0 and 0 or a/b)
      --print("a " .. a .. " b " .. b .. " R " .. recalls[i])
      local a = precision_recall[i].truepos
      local b = precision_recall[i].truepos + precision_recall[i].falsepos
      precisions[i] = (a==0 and b==0 and 0 or a/b)
      --print("a " .. a .. " b " .. b .. " P " .. precisions[i] .. " fp " .. precision_recall[i].falsepos)
            
      macro_R = macro_R + ((recalls[i]==recalls[i]) and recalls[i] or 0)
      macro_P = macro_P + ((precisions[i]==precisions[i]) and precisions[i] or 0)
   end
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
   local micro_R = (a==0 and b==0 and 0 or a/b) 
   local a = _truepos
   local b = _truepos + _falsepos
   --print("a " .. a .. " b " .. b)
   local micro_P = (a==0 and b==0 and 0 or a/b)
   local micro_f1score = (2 * micro_R * micro_P) / (micro_R + micro_P)
   micro_f1score = micro_f1score==micro_f1score and micro_f1score or 0

   
   --print(data.relationhash)
   print("\t\tP\tR")
   for k,i in pairs(class_to_consider) do
      print("Class " .. i .. ":\t" .. string.format('%.2f',precisions[i]) .. "\t" .. string.format('%.2f',recalls[i])) 
   end

   print(confusion_matrix)

   return (macro_P or 0), (macro_R or 0), (macro_f1score or 0), cost, micro_P, micro_R, micro_f1score      
end
