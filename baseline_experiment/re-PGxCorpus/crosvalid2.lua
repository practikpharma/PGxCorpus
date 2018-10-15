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

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())
torch.manualSeed(os.time())

local params = cmd:parse(arg)
torch.setnumthreads(1)

local cmd = "find " .. params.loaddir .. " -name model-best-valid.bin"
local handle = io.popen (cmd, "r")
print(cmd)


local data, vdata, tdata
local tab_rel = {"isAssociatedWith", "isExplainedBy", "treats", "transports", "influences", "increases", "decreases", "causes", "metabolizes", "isEquivalentTo", "relation"}


local tab_res = {}
for i=1,#tab_rel do
   tab_res[ tab_rel[i] ] = {f1={}, precision={}, recall={}}
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
   end

   data = createdata(paramsModel)
   vdata = extract_data(data, paramsModel.validp, paramsModel.valids, true)
   tdata = extract_data(data, paramsModel.validp, paramsModel.valids, true)
   subtraindata = extract_data(data, paramsModel.validp, paramsModel.valids, false)

   
   local network = createnetworks(paramsModel,data)
   local net = f:readObject()
   network:loadnet(paramsModel, net)
   
   paramsModel.rundir = params.loaddir .. paramsModel.rundir:match("/([^/]+)$")

   print("================")
   print("tab_res")
   print(tab_res.decreases)
   local tab = test(network, tdata, paramsModel)
   print("tab")
   print(tab.decreases)
   for i=2,#tdata.relationhash do
      local rel = tdata.relationhash[i]
      print(rel)
      
      table.insert(tab_res[rel].f1, tab[rel].f1==tab[rel].f1 and tab[rel].f1 or 0)
      table.insert(tab_res[rel].precision, tab[rel].precision==tab[rel].precision and tab[rel].precision or 0)
      table.insert(tab_res[rel].recall, tab[rel].recall==tab[rel].recall and tab[rel].recall or 0)
   end
   table.insert(tab_res.macro.recall, tab.macro_avg.recall)
   table.insert(tab_res.macro.precision, tab.macro_avg.precision)
   table.insert(tab_res.macro.f1, tab.macro_avg.f1)
  


   
   
   _file = handle:read()
end


print(tab_res)

for i=2,#tdata.relationhash do
   local rel = tdata.relationhash[i] 
   
   local avg_p = torch.Tensor(tab_res[rel].precision):mean()
   local avg_r = torch.Tensor(tab_res[rel].recall):mean()
   local avg_f1 = torch.Tensor(tab_res[rel].f1):mean()
   local std_f1 = f_std(tab_res[rel].f1, avg_f1)
   print("p\t" .. string.format("%.2f",avg_p*100) .. "\tr\t" .. string.format("%.2f",avg_r*100) .. "\tf1\t" .. string.format("%.2f",avg_f1*100) .. " ( " .. string.format("%.2f",std_f1*100) .. " )\t" .. rel)
end

local avg_p = torch.Tensor(tab_res.macro.precision):mean()
local avg_r = torch.Tensor(tab_res.macro.recall):mean()
local avg_f1 = torch.Tensor(tab_res.macro.f1):mean()
local std_f1 = f_std(tab_res.macro.f1, avg_f1)
print("p\t" .. string.format("%.2f",avg_p*100) .. "\tr\t" .. string.format("%.2f",avg_r*100) .. "\tf1\t" .. string.format("%.2f",avg_f1*100) .. " ( " .. string.format("%.2f",std_f1*100) .. " )\t" .. "macro")
