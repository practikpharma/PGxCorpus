require 'nn'
require 'rnn'
--require 'ReverseTable'

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

function checklm(words, hash, params)
   local w = 'expression'
   
   local cat = words[hash[w]]
   
   local dists = {}
   if not params.lmsum then
      for i=1,params.nword do
	 table.insert(dists, {i, words[i]:dist(cat)})
      end
      table.sort(dists, function(a, b)
		    return a[2] < b[2]
      end)
   else
      local cosdis = nn.CosineDistance()
      for i=1,params.nword do
	 table.insert(dists, {i, cosdis:forward({words[i],cat})[1]})
      end
      table.sort(dists, function(a, b)
		    return a[2] > b[2]
      end)
   end
   print('[lm] check words closest from <' .. w .. '>' )
   for i=1,10 do
      print(string.format('[lm]  -- %s (%g)', hash[dists[i][1]], dists[i][2]))
   end
end

function get_par(params, lkts, dropout, data, fixe)
   local par = nn.ParallelTable()
   if params.dropout~=0 and (params.dp==1 or params.dp==3) then
      if fixe then
	 local lkt = lkts.words:clone()
	 local oldaccgradparameters = lkt.backwardUpdate
	 function lkt.updateParameters(self, input, gradOutput, scale)
	 end
	 function lkt.accUpdateGradParameters(self, input, gradOutput, scale)
	 end
	 function lkt.accGradParameters(self, input, gradOutput, scale)
	 end
	 local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkt):add(d))
      else
	 local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.words:clone('weight','bias')):add(d))
      end
      if params.tfsz~=0 then
	 local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.entitytags:clone('weight','bias')):add(d))
      end
      if params.pfsz~=0 then
	 local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.pos:clone('weight','bias')):add(d))
      end
      if params.rdfsz~=0 then
	 local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.relativedistance1:clone('weight','bias')):add(d))
	 local d = nn.Dropout(params.dropout); table.insert(dropout, d); par:add(nn.Sequential():add(lkts.relativedistance2:clone('weight','bias')):add(d))
      end
      if params.nestenttype~=0 then
	 for i=3,#data.entityhash do --not padding and Other
	    local d = nn.Dropout(params.dropout); table.insert(dropout, d)
	    par:add(nn.Sequential():add(lkts[data.entityhash[i]]:clone('weight', 'bias')):add(d))
	 end
      end

   else
      if fixe then
	 local lkt = lkts.words:clone()
	 local oldaccgradparameters = lkt.backwardUpdate
	 function lkt.updateParameters(self, input, gradOutput, scale)
	 end
	 function lkt.accUpdateGradParameters(self, input, gradOutput, scale)
	    --print("hup")
	 end
	 function lkt.accGradParameters(self, input, gradOutput, scale)
	    --print("hop")
	 end
	 par:add(lkt)
      else
	 par:add(lkts.words:clone('weight','bias'))
      end
      if params.tfsz~=0 then par:add(lkts.entitytags:clone('weight','bias')) end
      if params.pfsz~=0 then par:add(lkts.pos:clone('weight','bias')) end
      if params.rdfsz~=0 then par:add(lkts.relativedistance1:clone('weight','bias')):add(lkts.relativedistance2:clone('weight','bias')) end
      if params.nestenttype~=0 then
	 for i=3,#data.entityhash do --not padding and Other
	    par:add(lkts[data.entityhash[i]]:clone('weight', 'bias')) end
      end
   end
   par:add(lkts.entities:clone('weight','bias'))
   
   return par
end
   
function createnetworks(params, data)
   local lkts = {}
   local words = nn.LookupTable(params.nword, params.wfsz)
   lkts.words = words
   local entities = nn.LookupTable(5, params.efsz)
   lkts.entities = entities
   --1:Padding 2:Other 3:entity1 4:entity2 5:node
   local entitytags, deptypes, relativedistance1, relativedistance2
   if params.tfsz~=0 then
      entitytags = nn.LookupTable(#data.entityhash, params.tfsz)
      lkts.entitytags = entitytags
   end
   if params.pfsz~=0 then
      pos = nn.LookupTable(#data.poshash, params.pfsz)
      lkts.pos = pos
   end
   if params.rdfsz~=0 then
      relativedistance1 = nn.LookupTable(300, params.rdfsz)
      relativedistance2 = nn.LookupTable(300, params.rdfsz)
      lkts.relativedistance1 = relativedistance1
      lkts.relativedistance2 = relativedistance2
   end
   if params.nestenttype~=0 then
      for i=3,#data.entityhash do --not padding and Other
	 lkts[data.entityhash[i]] = nn.LookupTable(3, params.nestenttype)
      end
   end
   
   if params.lm then
      print('loading lm')
      local wordhash = data.wordhash
      local lmdir = "embeddings/"
      local lmhash, s, f

      lmhash = loadhash(lmdir .. "/target_words.txt")
      s = string.format(lmdir .. '/words_%s.txt', params.wfsz)
      --s = string.format(lmdir .. '/words_%s.bin', params.wfsz)
      f = torch.DiskFile(s)
      
      print(string.format('[lm] %d words in the lm -- vs %d in the vocabulary', #lmhash, #wordhash))
      print("loading " .. s)
      if false then
	 local toto = f:readFloat(#lmhash * params.wfsz)
	 local fbis = torch.DiskFile(string.format('/home/joel/mobius/pubMedCorpora/request1/words_%s.bin', params.wfsz), "w")
	 fbis:writeObject(toto)
	 fbis:close()
	 exit()
      end

      local lmrepr = torch.Tensor(f:readFloat(#lmhash * params.wfsz), 1, #lmhash, -1, params.wfsz, -1)
      --local lmrepr = torch.Tensor(f:readObject(), 1, #lmhash, -1, params.wfsz, -1)
      
      if params.norm then
	 print("normalising embeddings")
	 local mean = lmrepr:mean()
	 local std = lmrepr:std()
	 lmrepr:add(-mean)
	 lmrepr:div(std)
      end
      f:close()
      
      local nknownword = 0

      for wordidx=1, params.nword do
	 local wrd = wordhash[wordidx]
	 if not wrd then
	    print(wordidx)
	    print(#wordhash)
	    print(wordhash[wordidx+1])
	 end
	 while wrd:match('%d%.%d') do
	    wrd = wrd:gsub('%d%.%d', '0')
	 end
	 while wrd:match('%d,%d') do
	    wrd = wrd:gsub('%d,%d', '0')
	 end
	 wrd = wrd:gsub('%d', '0')
	 local lmidx = lmhash[wrd]
	 if lmidx then
	    words.weight[wordidx]:copy(lmrepr[lmidx])
	    nknownword = nknownword + 1
	 end
      end
      print(string.format('[lm] %d known words (over %d in the vocabulary)', nknownword, params.nword))
      print('done')
      collectgarbage()
      
      checklm(words.weight, data.wordhash, params)
      
   end
   
   local network

   if params.dp==2 or params.dp==3 then error("dp==2 not authorized for arch==6") end
   
   local wszs = {}
   network = nn.Sequential()
   local net = nn.ConcatTable()
   local fsz = params.wfsz + params.efsz + params.tfsz + params.pfsz + (2*params.rdfsz)
   if params.nestenttype~=0 then
      for i=3,#data.entityhash do fsz = fsz + params.nestenttype end
   end
   print(fsz)
   local dropout = {}
   for i=1,#params.wszs do
      if params.tfsz>0 then
	 assert(data.wordhash["PADDING"]==data.entityhash["PADDING"] and data.entityhash["PADDING"]==data.entityhash2["PADDING"])
      end
      local pad = (params.wszs[i]-1)/2
      wszs[i] = nn.Sequential()
      local padding = nn.MapTable()
      local p = nn.Sequential()
      p:add(nn.Padding(1,-pad,1,data.wordhash["PADDING"]))
      p:add(nn.Padding(1,pad,1,data.wordhash["PADDING"]))
      padding:add(p)
      wszs[i]:add(padding)
      

      local channels = nn.ConcatTable()
      channels:add( nn.Sequential():add( get_par(params, lkts, dropout, data)):add(nn.JoinTable(2)):add( nn.TemporalConvolution(fsz, params.nhu[1], params.wszs[i])))
      if params.channels>1 then
	 channels:add( nn.Sequential():add( get_par(params, lkts, dropout, data, true)):add(nn.JoinTable(2)):add( nn.TemporalConvolution(fsz, params.nhu[1], params.wszs[i])))
      end
      if params.channels>2 then error("not implemented") end
      wszs[i]:add(channels)
      wszs[i]:add( nn.CAddTable() )
      
      
      --wszs[i]:add( nn.TemporalConvolution(fsz, params.nhu[1], params.wszs[i]) )
      wszs[i]:add( nn.HardTanh() )--non linearity
      wszs[i]:add( nn.Max(1) )
      net:add(wszs[i])
   end
   
   network:add(net)
   network:add( nn.JoinTable(1))

   --print(network)
   --exit()
   
   network = {network=network}
   
   network.scorer = nn.Sequential()
   network.scorer:add(nn.Linear(#wszs * params.nhu[1], #data.relationhash))
   network.scorer:add(nn.LogSoftMax())
   
   network.dropout = dropout
   if params.dropout~=0 and params.dp~=2 then
      function network.dropout:training()
	 for i=1,#self do self[i]:training() end
      end
      function network.dropout:evaluate()
	 for i=1,#self do self[i]:evaluate() end
      end
   end
   
   function network:forward(input)
      self.rep = self.network:forward(input)
      return self.scorer:forward(self.rep)
   end
   
   function network:backward(input,grad)
      print("grad")
      print(grad)
      local gradrep = self.scorer:backward(self.rep, grad)
      self.network:backward(input, gradrep)
   end

   function network:backwardUpdate(input, grad, lr)
      local gradrep = self.scorer:backwardUpdate(self.rep, grad, lr)
      self.network:backwardUpdate(input, gradrep, lr)
   end

   function network:zeroGradParameters()
      self.network:zeroGradParameters()
      self.scorer:zeroGradParameters()
   end
   
   function network:updateParameters(lr)
      self.network:updateParameters(lr)
      self.scorer:updateParameters(lr)
   end

   function network:training()
      self.network:training()
      self.scorer:training()
   end
   
   function network:evaluate()
      self.network:evaluate()
      self.scorer:evaluate()
   end
   
   network.save = {}
   table.insert(network.save, network.network)
   table.insert(network.save, network.scorer)
   
   
   if false then
      local input = {torch.Tensor({4,2,3}), torch.Tensor({4,2,3})}
      print("net")
      print(net)
      print("input")
      print(input)
      local output = network:forward(input, "full")
      print("output")
      print(output)
      -- print(output[1][1])
      -- print(output[1][2])
      -- print(output[2][1])
      -- print(output[2][2])
      exit()
   end
   
   function network:printnet()
      
   end
   
   function network:getnetsave(params)
      local res = {}
      for i=1,1 do
	 local p, g = network.save[i]:parameters()--clone("weight", "bias"):
	 table.insert(res, p)
      end
      res[2] = self.scorer:clone("weight", "bias"):parameters()
      return res
   end
   
   function network:loadnet(params, net)
      local c=1
      --print(net)
      for i=1,1 do
	 local oldparameters, oldgrads = network.save[i]:parameters()
	 --if i==2 then oldparameters = oldgrads end
	 for j=1,#oldparameters do
	    oldparameters[j]:copy( net[i][j] )
	 end
      end
      local oldparameters = network.scorer:parameters()
      --print(oldparameters)
      for j=1,#oldparameters do
	 oldparameters[j]:copy( net[2][j] )
      end
   end
   
   return network
end
