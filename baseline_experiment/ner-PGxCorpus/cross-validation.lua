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
require("test")
require 'TagInferenceBIOES'

cmd = torch.CmdLine()

cmd:text()
cmd:text('crosvalid')
cmd:text()
cmd:text()
cmd:text('Misc options:')
cmd:option('-loaddir', '', 'models to load')
cmd:option('-maxnet', 10, 'max number of network to load')
--cmd:option('-minnet', 1, 'max number of network to load')
cmd:option('-mobius', false, 'mobius')
cmd:option('-optnet', '', 'select networks with a given option')
cmd:option('-testcorpus',  '', 'corpus to teston')
cmd:option('-hierarchy', false, "consider entity hierarchy at test time")
cmd:option('-softmatch', false, "soft match at test time")
cmd:text()

math.randomseed(os.time())
torch.manualSeed(os.time())

local params = cmd:parse(arg)
local testfunction = test

torch.setnumthreads(1)

local cmd = "find " .. params.loaddir .. " -name model-best.bin"
local handle = io.popen (cmd, "r")
print(cmd)

local data, vdata, tdata

local tab_ent = {"Phenotype", "Disease", "Pharmacokinetic_phenotype", "Pharmacodynamic_phenotype", "Genomic_factor", "Genomic_variation", "Gene_or_protein", "Limited_variation", "Haplotype", "Chemical"}

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
   if _file:match(params.optnet) then
      nnetwork = nnetwork + 1

      print(_file .. " net n°" .. nnetwork)
      local f = torch.DiskFile(_file):binary()
      paramsModel = f:readObject()
      
      data = createdata(paramsModel)
      vdata = extract_data(data, paramsModel.validp, paramsModel.valids)
      tdata = extract_data(data, paramsModel.validp, paramsModel.valids)

      local networks = f:readObject()
      local tagger = f:readObject()
      
      print("now testing net " .. nnetwork)

      paramsModel.rundir = params.loaddir .. paramsModel.rundir:match("/([^/]+)$")
      paramsModel.hierarchy = params.hierarchy
      paramsModel.softmatch = params.softmatch
      print("================================ tdata")
      local tab = testfunction(networks, tagger, paramsModel, tdata, "train")
      for i=1,#tdata.entityhash do
	 local ent = tdata.entityhash[i] 
	 table.insert(tab_res[ent].f1, tab[ent].f1==tab[ent].f1 and tab[ent].f1 or 0)
	 table.insert(tab_res[ent].precision, tab[ent].precision==tab[ent].precision and tab[ent].precision or 0)
	 table.insert(tab_res[ent].recall, tab[ent].recall==tab[ent].recall and tab[ent].recall or 0)
      end
      table.insert(tab_res.macro.recall, tab.macro_avg.recall)
      table.insert(tab_res.macro.precision, tab.macro_avg.precision)
      table.insert(tab_res.macro.f1, tab.macro_avg.f1)
  
   end
   _file = handle:read()
end

print(tab_res)

for i=1,#tdata.entityhash do
   local ent = tdata.entityhash[i] 
   
   local avg_p = torch.Tensor(tab_res[ent].precision):mean()
   local avg_r = torch.Tensor(tab_res[ent].recall):mean()
   local avg_f1 = torch.Tensor(tab_res[ent].f1):mean()
   local std_f1 = f_std(tab_res[ent].f1, avg_f1)
   --print("p\t" .. string.format("%.2f",avg_p*100) .. "\tr\t" .. string.format("%.2f",avg_r*100) .. "\tf1\t" .. string.format("%.2f",avg_f1*100) .. " ( " .. string.format("%.2f",std_f1*100) .. " )\t" .. ent)
   print("" .. string.format("%.2f",avg_p*100) .. " & " .. string.format("%.2f",avg_r*100) .. " & " .. string.format("%.2f",avg_f1*100) .. " (" .. string.format("%.2f",std_f1*100) .. ")\t" .. ent)
end

local avg_p = torch.Tensor(tab_res.macro.precision):mean()
local avg_r = torch.Tensor(tab_res.macro.recall):mean()
local avg_f1 = torch.Tensor(tab_res.macro.f1):mean()
local std_f1 = f_std(tab_res.macro.f1, avg_f1)
--print("p\t" .. string.format("%.2f",avg_p*100) .. "\tr\t" .. string.format("%.2f",avg_r*100) .. "\tf1\t" .. string.format("%.2f",avg_f1*100) .. " ( " .. string.format("%.2f",std_f1*100) .. " )\t" .. "macro")
print("" .. string.format("%.2f",avg_p*100) .. " & " .. string.format("%.2f",avg_r*100) .. " & " .. string.format("%.2f",avg_f1*100) .. " (" .. string.format("%.2f",std_f1*100) .. ")\t" .. "macro")
