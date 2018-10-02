require 'nn'

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

local function countparams(net, params)
   local ps = net:parameters()
   local n = 0
   for _,p in ipairs(ps) do
      n = n + p:nElement()
   end
   print("number of parameters : " .. n)
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


function createnetworks(params, data)
   local words = nn.LookupTable(params.nword, params.wfsz)
   local caps

   if params.lm then
      print('loading lm')
      local wordhash = data.wordhash

      local lmdir = "embeddings/"
      
      local lmhash, s, f
      --lmhash = loadhash("/home/joel/Bureau/loria/code/hpca/target_words.txt")
      --s = string.format('/home/joel/Bureau/loria/code/hpca/words_%s.txt', params.wfsz)
      --lmhash = loadhash("/home/joel/mobius/pubMedCorpora/request1/target_words.txt")
      --s = string.format('/home/joel/mobius/pubMedCorpora/request1/words_%s.txt', params.wfsz)

      lmhash = loadhash(lmdir .. "/target_words.txt")
      s = string.format(lmdir .. '/words_%s.txt', params.wfsz)
      f = torch.DiskFile(s)
      
      print(string.format('[lm] %d words in the lm -- vs %d in the vocabulary', #lmhash, #wordhash))
      print("loading " .. s)
      local lmrepr = torch.FloatTensor(f:readFloat(#lmhash * params.wfsz), 1, #lmhash, -1, params.wfsz, -1):double()
      
      
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

      checklm(words.weight, data.wordhash, params)

      if params.divlm~=0 then
	 words.weight:div(params.divlm)
	 checklm(words.weight, data.wordhash, params)
      end
      
   end
   
   local features = nn.ParallelTable()
   local fsz = 0
   --words
   features:add( words )
   fsz = fsz + params.wfsz
   --lower layer labels 
   local labels = nn.LookupTable(#data.labelhash, params.lfsz)
   features:add(labels)
   fsz = fsz + params.lfsz
   --pubtatorhash
   if params.pfsz~=0 then
      local pubtators = nn.LookupTable(#data.pubtatorhash, params.pfsz)
      features:add(pubtators)
      fsz = fsz + params.pfsz
   end

   local scorer
   if params.nhu2~=0  then
      scorer = nn.Sequential()
      scorer:add(nn.Linear(params.nhu, params.nhu2))
      scorer:add(nn.HardTanh())
      scorer:add(nn.Linear(params.nhu2 , params.nlabel))
   else
      scorer = nn.Linear(params.nhu2~=0 and params.nhu2 or params.nhu, params.nlabel)
   end
   local networks = {}

   local net = nn.Sequential()
   net:add( features )
   net:add( nn.JoinTable(2) )
   if params.dropout~=0 then
      local dropout = nn.Dropout(params.dropout)
      net:add(dropout)
      net.dropout = dropout
   end
   net:add( nn.TemporalConvolution(fsz, params.nhu, params.wsz) )
   net:add( nn.HardTanh() )
   net:add( scorer )
   networks = net
   
   networks.words = words
   networks.fsz = fsz
   networks.scorer = scorer
   networks.features = features
   
   countparams(networks, params)
   
   return networks

end
