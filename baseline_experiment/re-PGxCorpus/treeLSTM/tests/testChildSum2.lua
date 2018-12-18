require "nn"
require "nngraph"

treelstm = {}
include("../util/Tree.lua")
include('../layers/CRowAddTable.lua')
include('../models/LSTM.lua')
include('../models/TreeLSTM.lua')
include('../models/ChildSumTreeLSTM.lua')

function treelstm.Tree:print_tree(tab)
   print(tab .. self.idx)
   for i=1,#self.children do
      self.children[i]:print_tree(tab .. "\t")
   end
end

-- local tree = {1, 2, 1, 2}

-- local j = 1
-- local reps = {}
-- while j<#tree do
--    local size = tree[j]
--    j = j+1
--    local head = treelstm.Tree()
--    head.idx = tree[j]
--    j = j + 1
--    for k=1,size-1 do
--       if tree[j]<1000 then
-- 	 local son = treelstm.Tree()
-- 	 son.idx = tree[j]
-- 	 head:add_child(son)
--       else
-- 	 head:add_child(reps[ tree[j]-1000 ])
--       end
--       j = j + 1
--    end
--    table.insert(reps, head)
-- end

--local head = reps[#reps]

local idx = 1
local head = treelstm.Tree()
head.idx = idx; idx = idx + 1
for i=1,math.random(3) do
   local son = treelstm.Tree()
   son.idx = idx; idx = idx + 1
   head:add_child(son)
   for j=1, math.random(3) do
      local grandson = treelstm.Tree()
      grandson.idx = idx; idx = idx + 1
      son:add_child(grandson)
   end
end
   
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
   in_dim = 10,
   mem_dim = 20,
   gate_output = true,
}

local net = treelstm.ChildSumTreeLSTM(treelstm_config)
      
local input = torch.rand(idx,10)

print(head.idx)

local output = net:forward(head,input)

print(output)

net:backward(head, input, output)
print("done")


net:forward(head,input)
local output = net:backward(head, input, output)
print("done")
