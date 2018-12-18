treelstm = {}
include("../util/Tree.lua")

function treelstm.Tree:print_tree(tab)
   print(tab .. self.idx .. (self.tag and " " .. self.tag or ""))
   for i=1,#self.children do
      self.children[i]:print_tree(tab .. "\t")
   end
end


function treelstm.Tree:LCA(a, b, entities, default_tag, inpath_tag)
   self.tag=default_tag
   local fa, fb, found
   for i=1,#self.children do
      _fa, _fb, _found = self.children[i]:LCA(a,b,entities,default_tag, inpath_tag)
      found = _found or found
      fa = _fa or fa
      fb = _fb or fb
   end
   if not found then
      fa = fa or (entities[self.idx]==a)
      fb = fb or (entities[self.idx]==b)
      if fa or fb then self.tag=inpath_tag end
   end
   return fa, fb, (fa and fb)
end


--building tree
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
local entities = torch.Tensor(12):fill(2)
entities[2]=3
entities[3]=3
entities[7]=4
head:LCA(3,4,entities,"N","P")
head:print_tree("")
