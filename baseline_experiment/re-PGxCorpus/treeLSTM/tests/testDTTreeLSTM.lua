require "nn"
require "nngraph"

torch.manualSeed(1)

treelstm = {}
include("../util/Tree.lua")
include('../layers/CRowAddTable.lua')
include('../models/LSTM.lua')
include('../models/TreeLSTM.lua')
include('../models/DTTreeLSTM.lua')
include('../layers/ConditionedLinear.lua')
include('../layers/ConditionedLinear2.lua')

function treelstm.Tree:print_tree(tab)
   print(tab .. self.idx .. " " .. self.tag)
   io.write(tab .. "{")
   for i = 1,#self.sontags do io.write(self.sontags[i] .. " ") end
   io.write("}\n")
   for i=1,#self.children do
      self.children[i]:print_tree(tab .. "\t")
   end
end

function treelstm.Tree:set_tab_sontags(leaftag)
   self.sontags = {}
   if #self.children==0 then
      table.insert(self.sontags, leaftag)
   else
      for i=1,#self.children do
	 table.insert(self.sontags, self.children[i].tag)
	 self.children[i]:set_tab_sontags(leaftag)
      end
   end
end

local tags = {1,2,3}

local idx = 1
local head = treelstm.Tree()
head.idx = idx; idx = idx + 1
head.tag = tags[math.random(#tags)]
for i=1,math.random(3) do
   local son = treelstm.Tree()
   son.idx = idx; idx = idx + 1
   son.tag = tags[math.random(#tags)]
   head:add_child(son)
   for j=1, math.random(3) do
      local grandson = treelstm.Tree()
      grandson.idx = idx; idx = idx + 1
      grandson.tag = tags[math.random(#tags)]
      son:add_child(grandson)
   end
end
   
head:set_tab_sontags(1)
head:print_tree("")

-- share module parameters
function share_params(cell, src)
   if torch.type(cell) == 'nn.gModule' then
      for i = 1, #cell.forwardnodes do
	 local node = cell.forwardnodes[i]
	 if node.data.module then
	    node.data.module:share(src.forwardnodes[i].data.module,
					 'weight', 'bias', 'gradWeight', 'gradBias')
	 end
      end
   elseif torch.isTypeOf(cell, 'nn.Module') then
      cell:share(src, 'weight', 'bias', 'gradWeight', 'gradBias')
   else
      error('parameters cannot be shared for this input')
   end
end

local treelstm_config = {
   in_dim = 3,
   mem_dim = 5,
   gate_output = true,
   tags = tags
}

local net = treelstm.DTTreeLSTM(treelstm_config)
    

local input = torch.rand(idx-1,treelstm_config.in_dim)
local output = net:forward(head,input)
local target = {}
table.insert(target, torch.rand(output[1]:size()))
table.insert(target, torch.rand(output[2]:size()))

local criterion = nn.MSECriterion()

local cost = criterion:forward(output[2], target[2])
local output = net:forward(head,input)
local cost = criterion:forward(output[2], target[2])

local epsilon = 0.000001

local weight, gradWeight = net:parameters()
print(weight)

--finite gradient method for weight and bias
print("============================= weight =============================")
for i=1,#weight do
   local w,gw = weight[i]:view(weight[i]:nElement()), gradWeight[i]:view(gradWeight[i]:nElement())
   for k=1,w:size(1) do
      --print(i .. " " .. k)
      local back = w[k]
      w[k] = w[k] + epsilon
      
      local output = net:forward(head, input)
      local cost2 = criterion:forward(output[2], target[2])
      local grad = {}
      grad[1] = output[1]:clone():zero()
      grad[2] = criterion:backward(output[2], target[2])
      
      local deriv = (cost2 - cost) / epsilon
      
      net:zeroGradParameters()
      local gradinput = net:backward(head, input, grad)
      
      --print(cost .. " " .. cost2)
      --print(deriv .. " " .. gw[k])
      print(deriv - gw[k])
      assert(math.abs(deriv - gw[k]) < 0.00001)
      
      w[k] = back
   end
end

print("============================= input =============================")
--finite gradient methode for input

for i=1,input:size(1) do
   for j=1,input:size(2) do
      local back = input[i][j]
      input[i][j] = input[i][j] + epsilon
      
      local output = net:forward(head, input)
      local cost2 = criterion:forward(output[2], target[2])
      local grad = {}
      grad[1] = output[1]:clone():zero()
      grad[2] = criterion:backward(output[2], target[2])
      
      local deriv = (cost2 - cost) / epsilon
      
      net:zeroGradParameters()
      local ginput = net:backward(head, input, grad)
      
      --print(cost .. " " .. cost2)
      --print(deriv .. " " .. ginput[i][j])
      print(deriv - ginput[i][j])
      assert(math.abs(deriv - ginput[i][j]) < 0.00001)
      
      input[i][j] = back
   end
end

