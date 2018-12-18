--[[


--]]

local DTTreeLSTM, parent = torch.class('treelstm.DTTreeLSTM', 'treelstm.TreeLSTM')

function DTTreeLSTM:__init(config)
   parent.__init(self, config)
   self.gate_output = config.gate_output
   if self.gate_output == nil then self.gate_output = true end
   self.dropout = config.dropout --add joel
   self.optim = config.optim
   
   -- a function that instantiates an output module that takes the hidden state h as input
   self.output_module_fn = config.output_module_fn
   self.criterion = config.criterion
   self.tags = config.tags and config.tags or error("config.tags needed")

   -- composition module
   self.composer = self:new_composer()
   self.composers = {}
   
   -- output module
   self.output_module = self:new_output_module()
   self.output_modules = {}
   
end


function DTTreeLSTM:new_composer()
   local input = nn.Identity()()
   local sontags = nn.Identity()()
   local child_c = nn.Identity()()
   local child_h = nn.Identity()()
   --local child_h_sum = nn.Sum(1)(child_h)

   local i = nn.Sigmoid()(
      nn.CAddTable(){
	 nn.Linear(self.in_dim, self.mem_dim)(input),
	 nn.Sum(1)(nn.ConditionedLinear(self.mem_dim, self.mem_dim, false, #self.tags, self.optim){child_h, sontags})
			 })
   local f = nn.Sigmoid()(
      treelstm.CRowAddTable(){
	 --nn.TemporalConvolution(self.mem_dim, self.mem_dim, 1)(child_h),
	 nn.Sum(2)(nn.ConditionedLinear2(self.mem_dim, self.mem_dim, false, #self.tags, self.optim){child_h, sontags}),
	 nn.Linear(self.in_dim, self.mem_dim)(input),
			 })
   local update = nn.Tanh()(
      nn.CAddTable(){
	 nn.Linear(self.in_dim, self.mem_dim)(input),
	 nn.Sum(1)(nn.ConditionedLinear(self.mem_dim, self.mem_dim, false, #self.tags, self.optim){child_h, sontags})
	 --nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
			   })
   local c = nn.CAddTable(){
      nn.CMulTable(){i, update},
      nn.Sum(1)(nn.CMulTable(){f, child_c})
			   }

   local h
   if self.gate_output then
      local o = nn.Sigmoid()(
	 nn.CAddTable(){
	    nn.Linear(self.in_dim, self.mem_dim)(input),
	    nn.Sum(1)(nn.ConditionedLinear(self.mem_dim, self.mem_dim, false, #self.tags, self.optim){child_h, sontags})
	    --nn.Linear(self.mem_dim, self.mem_dim)(child_h_sum)
			    })
      h = nn.CMulTable(){o, nn.Tanh()(c)}
   else
      h = nn.Tanh()(c)
   end

   if self.dropout then  --add joel
      local h_ = h  --add joel
      h = nn.Dropout(self.dropout)(h_) --add joel
   end  --add joel
   
   local composer = nn.gModule({input, child_c, child_h, sontags}, {c, h})
   --local composer = nn.gModule({input, child_c, child_h}, {c, h})
   if self.composer ~= nil then
      share_params(composer, self.composer)
   end
   
   return composer
end


function DTTreeLSTM:new_output_module()
   if self.output_module_fn == nil then return nil end
   local output_module = self.output_module_fn()
   if self.output_module ~= nil then
      share_params(output_module, self.output_module)
   end
   return output_module
end

function DTTreeLSTM:forward(tree, inputs)
  -- print("forward node " .. tree.idx .. " with " .. #tree.children .. " nodes")
   local loss = 0
   for i = 1, tree.num_children do
      local _, child_loss = self:forward(tree.children[i], inputs)
      loss = loss + child_loss
   end
   local child_c, child_h = self:get_child_states(tree)
   self:allocate_module(tree, 'composer')
   --print("DTTreeLSTM (node " .. tree.idx .. ")")
   --print(tree.sontags)
   tree.state = tree.composer:forward{inputs[tree.idx], child_c, child_h, tree.sontags}

   if self.output_module ~= nil then
      self:allocate_module(tree, 'output_module')
      tree.output = tree.output_module:forward(tree.state[2])
      if self.train and tree.gold_label ~= nil then
	 loss = loss + self.criterion:forward(tree.output, tree.gold_label)
      end
   end

   -- graph.dot(tree.composer.fg, 'MLP')
   -- io.read()d
   --print("done " .. tree.idx)
   return tree.state, loss
end

local grad_inputs = torch.Tensor()
function DTTreeLSTM:backward(tree, inputs, grad)
   grad_inputs:resizeAs(inputs)
   self:_backward(tree, inputs, grad, grad_inputs)
   return grad_inputs
end

function DTTreeLSTM:_backward(tree, inputs, grad, grad_inputs)
   --print("backward node " .. tree.idx)
   local output_grad = self.mem_zeros
   if tree.output ~= nil and tree.gold_label ~= nil then
      output_grad = tree.output_module:backward(
	 tree.state[2], self.criterion:backward(tree.output, tree.gold_label))
   end
   self:free_module(tree, 'output_module')
   tree.output = nil

   local child_c, child_h = self:get_child_states(tree)
   local composer_grad = tree.composer:backward(
      {inputs[tree.idx], child_c, child_h, tree.sontags},
      {grad[1], grad[2] + output_grad})
   self:free_module(tree, 'composer')
   tree.state = nil

   grad_inputs[tree.idx] = composer_grad[1]
   local child_c_grads, child_h_grads = composer_grad[2], composer_grad[3]
   for i = 1, tree.num_children do
      self:_backward(tree.children[i], inputs, {child_c_grads[i], child_h_grads[i]}, grad_inputs)
   end
end

function DTTreeLSTM:clean(tree)
   self:free_module(tree, 'composer')
   self:free_module(tree, 'output_module')
   tree.state = nil
   tree.output = nil
   for i = 1, tree.num_children do
      self:clean(tree.children[i])
   end
end

function DTTreeLSTM:parameters()
   local params, grad_params = {}, {}
   local cp, cg = self.composer:parameters()
   tablex.insertvalues(params, cp)
   tablex.insertvalues(grad_params, cg)
   if self.output_module ~= nil then
      local op, og = self.output_module:parameters()
      tablex.insertvalues(params, op)
      tablex.insertvalues(grad_params, og)
   end
   return params, grad_params
end

function DTTreeLSTM:get_child_states(tree)
   local child_c, child_h
   if tree.num_children == 0 then
      child_c = torch.zeros(1, self.mem_dim)
      child_h = torch.zeros(1, self.mem_dim)
   else
      child_c = torch.Tensor(tree.num_children, self.mem_dim)
      child_h = torch.Tensor(tree.num_children, self.mem_dim)
      for i = 1, tree.num_children do
	 child_c[i], child_h[i] = unpack(tree.children[i].state)
      end
   end
   return child_c, child_h
end
