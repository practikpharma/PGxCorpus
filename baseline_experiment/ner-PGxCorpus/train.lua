require('torch')
require('data')
require('network')
require('trepl')
require('nn')
require('test')
require 'TagInferenceBIOES'


function printw(words, hash)
   print('size : ' .. words:size(1))
   for i=1, words:size(1) do
      io.write(hash[words[i]] .. ' ')
   end
   io.write('\n')
end

cmd = torch.CmdLine()

cmd:text()
cmd:text('Chunk-based phrase prediction')
cmd:text()
cmd:text()
cmd:text('Misc options:')
cmd:option('-data', 'data/PGxCorpus', 'data directory')
cmd:option('-dropout', 0, 'add dropout')
cmd:option('-nword', 20000, 'dictionary size')
cmd:option('-iter', 100, 'max number of iterations')
cmd:option('-wsz', 5, 'window size')
cmd:option('-wfsz', 50, 'word feature size')
cmd:option('-tfsz', 0, 'tag feature size')
cmd:option('-nhu', 100, 'number of hidden units')
cmd:option('-nhu2', 0, 'number of hidden units')
cmd:option('-lr', 0.025, 'learning rate')
cmd:option('-lm', false, 'use language model')
cmd:option('-norm', false, 'normalize language model')
cmd:option('-divlm', 0, '')
cmd:option('-forward', false, 'use forward instead viterbi during training')
cmd:option('-hack', false, 'hack linear/convolution lr')
cmd:option('-zero', false, 'zero trans')
cmd:option('-dir', '.', 'subdirectory to save the stuff')
cmd:option('-seed', 1111, 'seed')
cmd:option('-restart', '', 'model to restart from')
cmd:option('-lrcrf', 0.025, 'learning rate for the crf')
cmd:option('-task', 'pgx', 'gm=gene mention')
cmd:option('-log', false, 'log file')
cmd:option('-maxload', 0, 'data size')
cmd:option('-tov', false, 'train on valid')
cmd:option('-pfsz', 20, 'pubtator tag feature size')
cmd:option('-lfsz', 20, 'previous layer tag feature size)')
cmd:option('-level1', 0, 'use pubtator tags as input (pubtator feature size)')
cmd:option('-validp', 10, 'training corpus proportion for the validation')
cmd:option('-valids', 1, 'sector extracted from training for the validation')
cmd:option('-hierarchy', false, "consider entity hierarchy at test time")
cmd:option('-softmatch', false, "soft match at test time")
cmd:option('-brat', false, "")
cmd:option('-levelmax', math.huge, 'level max. Caution: levels start at 0!')
cmd:option('-onlylabel', '{Phenotype=true, Disease=true, Pharmacokinetic_phenotype=true, Pharmacodynamic_phenotype=true, Genomic_factor=true, Genomic_variation=true, Gene_or_protein=true, Limited_variation=true, Haplotype=true, Chemical=true}', 'on labels given in option')
cmd:text()

--torch.setdefaulttensortype('torch.FloatTensor')

local params = cmd:parse(arg)

if true then
   torch.manualSeed(params.seed)
else
   torch.manualSeed(os.time())
end


print(params.onlylabel)
params.onlylabel = loadstring("return " .. params.onlylabel)()
print(params.onlylabel)

torch.setnumthreads(1)

local frestart
local rundir
local expidx = 0
if params.restart ~= '' then
   print(string.format('restarting from <%s>', params.restart))
   frestart = torch.DiskFile(params.restart):binary()
   local restart = params.restart
   params = frestart:readObject()
   params.restart = restart
   rundir = params.rundir
   for i=0,99 do
      expidx = i
      local fname = string.format('%s/log-%.2d', rundir, expidx)
      local f = io.open(fname)
      if f then
         print(string.format('<%s> found', fname))
         f:close()
      else
         break
      end
   end
else
   rundir = cmd:string('exp', params, {dir=true, loadnet=true, loadscorer=true, onlylabel=true})
   if params.dir ~= '.' then
      rundir = params.dir .. '/' .. rundir
   end
   
   params.rundir = rundir
   os.execute('mkdir -p ' .. rundir)
   params.currentiter = 0
end
if params.log then
   cmd:log(string.format('%s/log-%.2d', rundir, expidx), params)
end

if params.goldorpred==nil then params.goldorpred="gold" end


print("restart with : ")
print("luajit train.lua -restart " .. string.format('%s/model.bin', rundir):gsub("=", "\\="))



local testfunction = test

if params.hack then
   local oldaccgradparameters = nn.Linear.accGradParameters
   function nn.Linear.accGradParameters(self, input, gradOutput, scale)
      oldaccgradparameters(self, input, gradOutput, scale/( self.weight:size(2) ) )
   end

   local oldaccgradparameters = nn.TemporalConvolution.accGradParameters
   function nn.TemporalConvolution.accGradParameters(self, input, gradOutput, scale)
      oldaccgradparameters(self, input, gradOutput, scale/self.weight:size(2))
   end
end


--------------------------DATA------------------------------
local data = createdata(params, "train")
params.nlabel = #data.chunkhash
params.nword = math.min(params.nword, #data.wordhash)
params.ntags = params.tfsz~=0 and #data.taghash or nil

--computing statistics abous discontiguous and nested entities
for i=1,data.size do
   print(data.entities[i])
   print(data.entities[i][10].sons)
   
   io.read()
end

--computing max sentence size (in terms of words)
local maxsize=0
for i=1,data.size do
   if data.words[i]:size(1)>maxsize then maxsize = data.words[i]:size(1) end
end
-- for i=1,vdata.size do
--    if vdata.words[i]:size(1)>maxsize then maxsize = vdata.words[i]:size(1) end
-- end
params.sizemax = maxsize


--------------------------NETWORK---------------------------
local networks
local tagger

if frestart then
   print('reloading network')
   networks = frestart:readObject()
   tagger = frestart:readObject()
else
   local sizemax = 276
   print("creating network")
   networks = createnetworks(params, data)
   tagger = nn.TagInferenceBIOES(data.chunkhash, sizemax)
   if params.zero then
      print('zero trans')
      tagger.trans:zero()
      tagger.stop:zero()
   end
end

local networkssave = networks:clone("weight", "bias")

--cost
local fcost = io.open(string.format('%s/cost', rundir), 'a')
local fcostvalid = io.open(string.format('%s/cost-valid', rundir), 'a')
local fcosttest = io.open(string.format('%s/cost-test', rundir), 'a')

-- --------micro
-- --p
-- local fmicroprecisiontest = io.open(string.format('%s/micro-precision-test', rundir), 'a')
-- local fmicroprecisionvalid = io.open(string.format('%s/micro-precision-valid', rundir), 'a')
-- local fmicroprecision = io.open(string.format('%s/micro-precision', rundir), 'a')

-- --recall
-- local fmicrorecalltest = io.open(string.format('%s/micro-recall-test', rundir), 'a')
-- local fmicrorecallvalid = io.open(string.format('%s/micro-recall-valid', rundir), 'a')
-- local fmicrorecall = io.open(string.format('%s/micro-recall', rundir), 'a')

-- --f1
-- local fmicrof1test = io.open(string.format('%s/micro-f1-test', rundir), 'a')
-- local fmicrof1valid = io.open(string.format('%s/micro-f1-valid', rundir), 'a')
-- local fmicrof1 = io.open(string.format('%s/micro-f1', rundir), 'a')

-- --f1 label agnostic
-- local fnolabelf1test = io.open(string.format('%s/nolabel-f1-test', rundir), 'a')
-- local fnolabelf1valid = io.open(string.format('%s/nolabel-f1-valid', rundir), 'a')
-- local fnolabelf1 = io.open(string.format('%s/nolabel-f1', rundir), 'a')


local vdata = extract_data(data, params.validp, params.valids)


params.best = 0

collectgarbage()

-- print("testing")
-- local f1 = testfunction(networks, tagger, params, tdata, "testa")
-- print("score: " .. f1)


if params.restart~="" then

   params.brat = true
   --params.verbose = true
   local microf1, microprecision, microrecall, nolabelf1 = testfunction(networks, tagger, params, vdata, "valid")
   print("valid score: " .. microf1, microprecision, microrecall)
   print(res)
   params.verbose = nil
   
   -- local f1, res = testfunction(networks, tagger, params, tdata, "test")
   -- print("test score: " .. f1)
   -- print(res)
   
   -- tagger.trans:zero()
   -- tagger.stop:zero()

   -- print("test restart zero")
   -- local f1, res = testfunction(networks, tagger, params, vdata, "valid")
   -- print("valid score: " .. f1)
   -- print(res)
   
   -- local f1, res = testfunction(networks, tagger, params, tdata, "test")
   -- print("test score: " .. f1)
   -- print(res)
   
   --local f1 = testfunction(networks, tagger, params, data, "train")
   --print("train score: " .. f1)
   exit()
end


local input_perm = {}
for i=1,data.size do --for each sentence
   --print(data.labels)
   for j=1,#data.labels[i] do
      table.insert(input_perm, {i, j})
   end
end

--local f1 = testfunction(networks, tagger, params, data, "valid")

local input_labels = torch.Tensor()
local pad = (params.wsz-1)/2

print('now training')
for iter=1, params.iter do
   
   local skip = 0
   
   if params.dropout~=0 then
      networks.dropout.train=true
   end
   
   local timer = torch.Timer()
   
   local perm = torch.randperm(#input_perm)
   local cost = 0
   local nex = 0

   for i=1, #input_perm do
      local idx = perm[i] --i; print("caution!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")----
      local idx_sentence = input_perm[ idx ][1]
      local idx_level = input_perm[ idx ][2]
      --print(idx)
      
      local words = data.words[ idx_sentence ]
      
      local labels, pubtator, chunks
      if false then
	 if idx_level==1 then
	    labels = input_labels:resize(words:size()):fill(data.labelhash["O"])
	    for i=1,pad do
	       input_labels[i] = data.labelhash["PADDING"]
	       input_labels[ input_labels:size(1)-i+1 ] = data.labelhash["PADDING"]
	    end
	    pubtator = data.labels_pubtator[idx_sentence][1]
	    local sz = data.labels[idx_sentence][idx_level]:size(1)
	    chunks = data.labels[idx_sentence][idx_level]:narrow(1, pad + 1, sz-params.wsz+1)
	 else
	    labels = data.labels[idx_sentence][idx_level-1]
	    pubtator = input_labels:resize(words:size()):fill(data.pubtatorhash["O"])
	    for i=1,pad do
	       input_labels[i] = data.pubtatorhash["PADDING"]
	       input_labels[ input_labels:size(1)-i+1 ] = data.pubtatorhash["PADDING"]
	    end
	    local sz = data.labels[idx_sentence][idx_level]:size(1)

	    chunks = data.labels[idx_sentence][idx_level]:narrow(1, pad + 1, sz-params.wsz+1)
	 end
      else
	 if idx_level==1 then
	    labels = data.labels_input[idx_sentence][1]
	    pubtator = data.labels_pubtator[idx_sentence][1]
	    local sz = data.labels[idx_sentence][idx_level]:size(1)
	    chunks = data.labels[idx_sentence][idx_level]:narrow(1, pad + 1, sz-params.wsz+1)
	 else
	    labels = data.labels_input[idx_sentence][idx_level]
	    pubtator = input_labels:resize(words:size()):fill(data.pubtatorhash["O"])
	    for i=1,pad do
	       input_labels[i] = data.pubtatorhash["PADDING"]
	       input_labels[ input_labels:size(1)-i+1 ] = data.pubtatorhash["PADDING"]
	    end
	    local sz = data.labels[idx_sentence][idx_level]:size(1)
	    chunks = data.labels[idx_sentence][idx_level]:narrow(1, pad + 1, sz-params.wsz+1)
	 end
      end
      
      if false and idx_sentence==21 then
	 print("level:" .. idx_level)
	 print("words")
	 printw(words,data.wordhash)
	 print("label_input")
	 printw(labels, data.labelhash)
	 print("pubtator")
	 printw(pubtator, data.pubtatorhash)
	 print("prediction")
	 printw(chunks, data.labelhash)
	 io.read()
      end
      
      local T = words:size(1)-params.wsz+1
      local input = {}
      table.insert(input, words)
      table.insert(input, labels)
      if params.pfsz~=0 then
	 table.insert(input, pubtator)
      end
      
      -- print("words")
      -- printw(input[1], data.wordhash)
      -- print("labels")
      -- printw(input[2], data.labelhash)
      -- print("pubtator")
      -- printw(input[3], data.pubtatorhash)
      -- print("prediction")
      -- printw(chunks, data.chunkhash)
      
      
      -- print("\n===================================== INPUT TRAIN =====================================")
      -- printw(input[1], data.wordhash)
      -- printw(input[2], data.labelhash)
      -- printw(input[3], data.pubtatorhash)
      -- print("prediction")
      -- printw(chunks, data.chunkhash)
      -- io.read()
      
      
      local criterioninput = {}
      criterioninput = networks:forward(input)
      
      if false then --check grad
	 --networks params
	 local criterion = nn.MSECriterion()
	 local params, grads = networks:getParameters()
	 --print(params)
	 local epsilon = 0.001
	 for i=1,params:size(1) do
	    print("params[i]: " .. params[i])
	    print("grads[i]: " .. grads[i])
	    local backup = params[i]
	    local output1 = networks:forward(input)
	    local rand = torch.rand(output1:size())
	    local score1 = criterion:forward(output1, rand)
	    print("score1 : " .. score1)
	    params[i] = params[i] + epsilon
	    local output2 = networks:forward(input)
	    local score2 = criterion:forward(output2, rand)
	    print("score2 : " .. score2)
	    local g = criterion:backward(output2, rand)
	    networks:zeroGradParameters()
	    networks:backward(input, g)
	    local deriv = (score2 - score1) / epsilon
	    print("grad : " .. grads[i])
	    print("diff : " .. score2-score1)
	    print("deriv : " .. deriv)
	    print("error grad " .. math.abs(deriv - grads[i]))
	    if math.abs(deriv - grads[i]) > 0.01 then error(math.abs(deriv - grads[i])) end
	    params[i]=backup
	 end
      end
      

      if false then
	 local epsilon = 0.00001
	 for i=1,criterioninput:size(1) do
	    for j=1, criterioninput:size(2) do

	       print("words n " .. i .. " tag " .. data.chunkhash[j])
	       local backup = criterioninput[i][j]
	       local score1, path1 = tagger:forward_max(criterioninput)

	       criterioninput[i][j] = criterioninput[i][j] + epsilon
	       tagger:zeroGradParameters()
	       tagger:zeroGradInput(criterioninput)
	       
	       local score2, path2 = tagger:forward_max(criterioninput)
	       
	       local _, path = tagger:forward_max(criterioninput)
	       for i=1,chunks:size(1) do
		  print(data.chunkhash[chunks[i]] .. "\t" .. data.chunkhash[path1[i]] .. "\t" .. data.chunkhash[path2[i]])
	       end

	       local total = 0
	       for i=1,path2:size(1) do
		  total = total+criterioninput[ i ][ path2[i] ]
	       end
	       print("total " .. total)
	       local total = 0
	       
	       --tagger:forward_correct(criterioninput, chunks)
	       --score2 = -score2
	       print("score1 : " .. score1)
	       print("score2 : " .. score2)
	       tagger:zeroGradInput(criterioninput)
	       tagger:zeroGradParameters()
	       tagger:backward_max(criterioninput, 1)
	       local g = tagger.gradInput
	       print("diff scores: " .. score2-score1)
	       local deriv = (score2 - score1) / epsilon
	       print(g:size())
	       --print(g)
	       print("grad : " .. g[i][j])
	       --print("diff : " .. score2-score1)
	       print("deriv : " .. deriv)
	       print("error grad " .. math.abs(deriv - g[i][j]))
	       if math.abs(deriv - g[i][j]) > 0.001 then error("error in forward-backward: " .. math.abs(deriv - g[i][j])) end
	       criterioninput[i][j] = backup
	       if g[i][j]~=0 then
		  io.read()
	       end
	       --io.read()
	    end
	 end
      end

      if false then
	 local epsilon = 0.00001
	 for i=1,criterioninput:size(1) do
	    for j=1, criterioninput:size(2) do
	       
	       print("words n " .. i .. " tag " .. data.chunkhash[j])
	       local backup = criterioninput[i][j]
	       local score1 = tagger:forward_logadd(criterioninput)

	       criterioninput[i][j] = criterioninput[i][j] + epsilon
	       tagger:zeroGradParameters()
	       tagger:zeroGradInput(criterioninput)
	       
	       local score2 = tagger:forward_logadd(criterioninput)
	       
	       --tagger:forward_correct(criterioninput, chunks)
	       --score2 = -score2
	       print("score1 : " .. score1)
	       print("score2 : " .. score2)
	       tagger:zeroGradInput(criterioninput)
	       tagger:zeroGradParameters()
	       tagger:backward_logadd(criterioninput, 1)
	       print("tata")
	       local g = tagger.gradInput
	       print("diff scores: " .. score2-score1)
	       local deriv = (score2 - score1) / epsilon
	       print(g:size())
	       --print(g)
	       print("grad : " .. g[i][j])
	       --print("diff : " .. score2-score1)
	       print("deriv : " .. deriv)
	       print("error grad " .. math.abs(deriv - g[i][j]))
	       if math.abs(deriv - g[i][j]) > 0.001 then error("error in forward-backward: " .. math.abs(deriv - g[i][j])) end
	       criterioninput[i][j] = backup
	       --io.read()
	    end
	 end
      end
      
      nex = nex + 1
      
      local path
      
      --print("=========")
      --print(words)
      --print(chunks)
      
      --print(data.chunkhash)
      --print(criterioninput)

      
      --c, path = tagger:forward_max(criterioninput)
      if params.forward then
	 cost = cost + tagger:forward_logadd(criterioninput)
      else
	 cost = cost + tagger:forward_max(criterioninput)
      end
      cost = cost - tagger:forward_correct(criterioninput, chunks)
      
      -- print(networks:get(1).output)
      -- print(networks:get(2).output)
      
      tagger:zeroGradParameters()
      tagger:zeroGradInput(criterioninput)
      
      tagger:backward_correct(criterioninput, chunks, -1)
      if params.forward then
	 tagger:backward_logadd(criterioninput, 1)
      else
	 tagger:backward_max(criterioninput, 1)
      end
      if not params.zero then tagger:updateParameters(params.lrcrf) end

      --tagger:toString(3)
      --_, path = tagger:forward_max(criterioninput)
      --print(path)
      --io.read()

      local g = tagger.gradInput
      networks:zeroGradParameters()
      networks:backward(input, g)
      networks:updateParameters(params.lr)
      --networks:backwardUpdate(input, g, params.lr)
      
      ::continue::
      
   end
   
   cost = cost/nex
   
   print("nb skipped: " .. skip)
   print(string.format('# current cost = %.5f', cost))
   print(string.format('# ex/s = %.2f [%d ex over %d processed -- %.4g%%]', data.size/timer:time().real, nex, data.size, nex/data.size*100))

   fcost:write(cost .. "\n")
   fcost:flush()

   -- print('saving: last model')
   -- local f = torch.DiskFile(string.format('%s/model.bin', rundir), 'w'):binary()
   -- f:writeObject(params)
   -- f:writeObject(networkssave)
   -- f:writeObject(tagger)
   -- f:close()

   local tab = testfunction(networks, tagger, params, data, "train")

   print("=============================================== data")
   for i=1,#data.entityhash do
      local ent = data.entityhash[i] 
      print(ent .. " " .. tab[ent].f1 .. " P " .. tab[ent].precision .. " r " .. tab[ent].recall)
      f = io.open(string.format('%s/' .. ent .. '-train-f1', rundir), 'a')
      f:write((tab[ent].f1==tab[ent].f1 and tab[ent].f1 or 0) .. "\n")
      f:close()

      f = io.open(string.format('%s/' .. ent .. '-train-precision', rundir), 'a')
      f:write((tab[ent].precision==tab[ent].precision and tab[ent].precision or 0) .. "\n")
      f:close()
      
      f = io.open(string.format('%s/' .. ent .. '-train-recall', rundir), 'a')
      f:write((tab[ent].recall==tab[ent].recall and tab[ent].recall or 0) .. "\n")
      f:close()
   end


   if true then
      local tab = testfunction(networks, tagger, params, vdata, "valid")
      
      print("=============================================== vdata")
      for i=1,#vdata.entityhash do
	 local ent = vdata.entityhash[i] 
	 print(ent .. " " .. tab[ent].f1)
	 f = io.open(string.format('%s/' .. ent .. '-valid-f1', rundir), 'a')
	 f:write((tab[ent].f1==tab[ent].f1 and tab[ent].f1 or 0) .. "\n")
	 f:close()
	 
	 f = io.open(string.format('%s/' .. ent .. '-valid-precision', rundir), 'a')
	 f:write((tab[ent].precision==tab[ent].precision and tab[ent].precision or 0) .. "\n")
	 f:close()
	 
	 f = io.open(string.format('%s/' .. ent .. '-valid-recall', rundir), 'a')
	 f:write((tab[ent].recall==tab[ent].recall and tab[ent].recall or 0) .. "\n")
	 f:close()
      end
   end
   
   
   -- print("train score: " .. microf1)
   -- fmicrof1:write(microf1 .. "\n")
   -- fmicrof1:flush()
   -- fmicroprecision:write(microprecision .. "\n")
   -- fmicroprecision:flush()
   -- fmicrorecall:write(microrecall .. "\n")
   -- fmicrorecall:flush()
   -- fnolabelf1:write(nolabelf1 .. "\n")
   -- fnolabelf1:flush()

   -- local microf1, microprecision, microrecall, nolabelf1 = testfunction(networks, tagger, params, vdata, "test")
   -- print("---------------------------------------------------test score: f1=" .. microf1 .. " P=" .. microprecision .. " R=" .. microrecall)
   -- fmicrof1valid:write(microf1 .. "\n")
   -- fmicrof1valid:flush()
   -- fmicroprecisionvalid:write(microprecision .. "\n")
   -- fmicroprecisionvalid:flush()
   -- fmicrorecallvalid:write(microrecall .. "\n")
   -- fmicrorecallvalid:flush()
   -- fnolabelf1valid:write(nolabelf1 .. "\n")
   -- fnolabelf1valid:flush()

   
   -- if true then
   --    print(microf1)
   --    print(params.best)
   --    if microf1>params.best then
   -- 	 params.best = microf1
   -- 	 print('saving: best model')
   -- 	 local f = torch.DiskFile(string.format('%s/model-best-valid.bin', rundir), 'w'):binary()
   -- 	 f:writeObject(params)
   -- 	 f:writeObject(networkssave)
   -- 	 f:writeObject(tagger)
   -- 	 f:close()
   --    end
      
   -- end
   -- if (iter-1)%10==0 then
   --    local f1 = testfunction(networks, tagger, params, data, "train")
   --    print("train score: " .. f1)
   --    ftrain:write(f1 .. "\n")
   --    ftrain:flush()
   -- end
end
