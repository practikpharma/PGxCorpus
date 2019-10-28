require("torch")
require("nn")
require("data")
require("network")

cmd = torch.CmdLine()

cmd:text()
cmd:text('Voting')
cmd:text()
cmd:text()
cmd:text('Misc options:')
cmd:option('-loaddir', '', 'models to load')
cmd:option('-maxnet', 10, 'max number of network to load')
cmd:option('-minnet', 1, 'max number of network to load')
cmd:option('-valid', false, 'test on valid (else, on test)')
cmd:option('-sort', false, 'take the best model instead of random')
cmd:option('-mobius', false, 'mobius')
cmd:option('-optnet', '', 'select networks with a given option')
cmd:option('-data', 'data/PGxCorpus', 'data directory')
cmd:option('-decode', false, 'decode the corpus (without testing)')
cmd:option('-brat', '', 'generate brat files')
cmd:option('-data', 'data/PGxCorpus', 'data directory')
cmd:text()

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
torch.manualSeed(os.time())

local params = cmd:parse(arg)

params.rundir = params.loaddir .. "/votingRes/" .. (params.sort and "sort" or "nosort") .. "/"
os.execute('mkdir -p ' .. params.rundir)
print(params.rundir)
cmd:log(string.format('%s/log', params.rundir), params)


local cmd = "find " .. params.loaddir .. " -name model-best-valid.bin"
print(cmd)
local handle = io.popen (cmd, "r")

local sort = {}

local nbnet = 0
_file = handle:read()
print(_file)
print("----------------------------------------------")
local paramsModel
while _file and nbnet<params.maxnet do
   if _file:match("exp,") and _file:match(params.optnet) then
      print(_file)
      if paths.filep(_file) then
	 local f = torch.DiskFile(_file):binary()
	 print("loading " .. _file)
	 paramsModel = f:readObject()
	 local tros = {}
	 if params.bestvalid then 
	    tros.sc = paramsModel.best
	    --tros.sc = paramsModel.bestleaf
	 else 
	 tros.sc = paramsModel.best 
	 end
	 tros.file = _file
	 table.insert(sort, tros)
	 f:close()
      end
   end
   _file = handle:read()
end

print(paramsModel)

paramsModel.mobius=params.mobius


print("\n\ntab sorted")
for k, v in pairs(sort) do
   for k1, v1 in pairs(v) do
      print(k1 .. " " .. v1)
   end
end
print("\n")


loadhash(paramsModel)

paramsModel.data = params.data
local data = createdata(paramsModel, params.decode)
if paramsModel.arch=='treelstm' then
   get_trees(data, paramsModel)
end

local currentmaxnet = params.minnet-1
while true do
   currentmaxnet = currentmaxnet + 1

   print("============================================================")
   print("===============" .. currentmaxnet .. " networks================")
   print("============================================================")
   
   
   if params.sort then
      table.sort(sort, function(a,b) return a.sc > b.sc end)
   else      
      local tab = {}
      local perm = torch.randperm(#sort)
      for i=1,#sort do
	 tab[i] = sort[perm[i]]
      end
      sort = tab
   end

   local networks = {}

   print("\n\n--------------loading networks----------------")
   for i=1, currentmaxnet do
      local f = torch.DiskFile(sort[i].file):binary()
      print("using " .. sort[i].file)
      print("score " .. sort[i].sc)
      paramsModel = f:readObject()
      paramsModel.mobius=params.mobius
      --print(paramsModel)
      nbnet = nbnet + 1
      local network = createnetworks(paramsModel,data)
      local net = f:readObject()
      network:loadnet(paramsModel, net)

      table.insert(networks, network)
      f:close()
      file = handle:read()
   end
   print("----------------done----------------------------\n\n")
   params.nbnet = nbnet
   params.bestf1 = 0


   if params.decode then
      dofile("decode.lua")
      print("lets decode")
      paramsModel.brat = params.brat
      --print(data.entities[197])
      --exit()
      decode(networks, data, paramsModel)
      print("decoding done") 
      exit()      
   else
      --exit()
      dofile("testVoting.lua")
      
      os.execute("mkdir -p " .. params.rundir .. currentmaxnet)

      paramsModel.brat = params.brat
      local tab_return = testVoting(networks, data, paramsModel)
      print("Test_macro: " .. tab_return.macro_avg.precision .. " " .. tab_return.macro_avg.recall .. " " .. tab_return.macro_avg.f1)
      print("Test_micro: " .. tab_return.micro_avg.precision .. " " .. tab_return.micro_avg.recall .. " " .. tab_return.micro_avg.f1)

   end
   
   collectgarbage()
   if currentmaxnet==params.maxnet then
      break
   end
end

print("done")
