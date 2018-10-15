function f_std(tab, avg)
   local sum = 0
   for i=1,#tab do
      sum = sum + math.pow((tab[i]-avg), 2)
   end
   sum = sum/#tab
   return math.sqrt(sum)
end

require("torch")
require("nn")
require("data")
require("network")
require("nngraph")
require("test")

cmd = torch.CmdLine()

cmd:text()
cmd:text('crosvalid')
cmd:text()
cmd:text()
cmd:text('Misc options:')
cmd:option('-loaddir', '', 'models to load')
cmd:option('-maxnet', 10, 'max number of network to load')
--cmd:option('-minnet', 1, 'max number of network to load')
cmd:option('-optnet', '', 'select networks with a given option')
cmd:option('-bestvalid', false, '')
cmd:text()

math.randomseed(os.time())
torch.manualSeed(os.time())

local params = cmd:parse(arg)
torch.setnumthreads(1)

local cmd = "find " .. params.loaddir .. " -name model-best.bin"
local handle = io.popen (cmd, "r")
print(cmd)


local data, vdata, tdata
local tab_rel = {"isAssociatedWith", "isExplainedBy", "treats", "transports", "influences", "increases", "decreases", "causes", "metabolizes", "isEquivalentTo", "relation"}


local tab_res = {}
for i=1,#tab_ent do
   tab_res[ tab_ent[i] ] = {f1={}, precision={}, recall={}}
end
tab_res.macro = {f1={}, precision={}, recall={}}


local nbnet = 0
_file = handle:read()
print("----------------------------------------------")
local paramsModel
local nnetwork = 0
while _file and nnetwork<params.maxnet do
   nnetwork = nnetwork + 1
   
   print(_file .. " net n°" .. nnetwork)
   local f = torch.DiskFile(_file):binary()
   paramsModel = f:readObject()

   if nnetwork==1 then
      loadhash(paramsModel)
      data = createdata(paramsModel)
      vdata = extract_data(data, params.validp, params.valids, true)
      tdata = extract_data(data, params.validp, params.valids, true)
      subtraindata = extract_data(data, params.validp, params.valids, false)
   end

   
   local network = createnetworks(paramsModel,datas)
   local net = f:readObject()
   network:loadnet(paramsModel, net)
   
   paramsModel.rundir = params.loaddir .. paramsModel.rundir:match("/([^/]+)$")

   local tab = test(network, tdata, params)
   print("Test_macro: " .. tab.macro_avg.precision .. " " .. tab.macro_avg.recall .. " " .. tab.macro_avg.f1)
   print("Test_micro: " .. tab.micro_avg.precision .. " " .. tab.micro_avg.recall .. " " .. tab.micro_avg.f1)


   
   
   _file = handle:read()
end


