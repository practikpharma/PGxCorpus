local ConditionedLinear, parent = torch.class('nn.ConditionedLinear', 'nn.Module')

function ConditionedLinear:__init(inputSize, outputSize, bias, ncond, optim)
   parent.__init(self)
   self.ncond = ncond or error("ncond must be entered")  
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize * self.ncond)
   self.gradWeight = torch.Tensor(outputSize, inputSize * self.ncond)
   self.weights = {}
   self.gradWeights = {}
   self.getWeight = function(i) 
      return self.weight:narrow(2,((i-1)*inputSize)+1, inputSize)
   end
   self.getGradWeight = function(i) 
      return self.gradWeight:narrow(2,((i-1)*inputSize)+1, inputSize)
   end

   -- for i=1,ncond do
   --    self.weights[i] = self.weight:narrow(2,((i-1)*inputSize)+1, inputSize)
   --    self.gradWeights[i] = self.gradWeight:narrow(2,((i-1)*inputSize)+1, inputSize)
   -- end
   
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end

   self.optim = optim
   -- self.temp_in, self.temp_out, self.to_fwd = {}, {}, torch.Tensor(ncond)
   -- if self.optim then
   --    for i=1,ncond do
   -- 	 self.temp_in[i]=torch.Tensor()
   -- 	 self.temp_out[i]=torch.Tensor()
   --    end
   -- end
   
   self:reset()
end

function ConditionedLinear:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function ConditionedLinear:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
      end
      if self.bias then
         for i=1,self.bias:nElement() do
            self.bias[i] = torch.uniform(-stdv, stdv)
         end
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end
   return self
end

local function updateAddBuffer(self, input)
   local nframe = input:size(1)
   self.addBuffer = self.addBuffer or input.new()
   if self.addBuffer:nElement() ~= nframe then
      self.addBuffer:resize(nframe):fill(1)
   end
end

function ConditionedLinear:updateOutput(inp)
   local input = inp[1]
   local cond = inp[2]
   -- print("input")
   -- print(input)
   -- print("cond")
   -- print(cond)
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then self.output:copy(self.bias) else self.output:zero() end
      self.output:addmv(1, self.getWeight( cond[1] ), input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      self.output:fill(0)
      updateAddBuffer(self, input)
      if self.optim then
	 if false then --pas intéressant (pas plus rapide voir plus lent)
	    self.to_fwd:fill(0)
	    --preparing self.temp_in and self.temp_out
	    for i=1,#cond do self.to_fwd[ cond[i] ] = self.to_fwd[ cond[i] ] + 1 end
	    for i=1,self.to_fwd:size(1) do
	       if self.to_fwd[i]~=0 then
		  --print("forward")
		  self.temp_in[i]:resize(self.to_fwd[i], input:size(2))
	       self.temp_out[i]:resize(self.to_fwd[i], self.weight:size(1))
	       end
	    end
	    --filling self.temp_in
	    self.to_fwd:fill(0)
	    for i=1,#cond do
	       self.to_fwd[ cond[i] ] = self.to_fwd[ cond[i] ] + 1 
	       --print(self.temp_in[ cond[i] ]:size())
	       self.temp_in[ cond[i] ][ self.to_fwd[cond[i]] ]:copy(input[i])
	    end
	    --forwarding self.temp_in 
	    for i=1,self.to_fwd:size(1) do
	       if self.to_fwd[i]~=0 then
		  --print(self.temp_out[i])
		  self.temp_out[ i ]:addmm(0, self.temp_out[i], 1, self.temp_in[i], self.getWeight(i):t())
	       end
	    end
	    --copy output
	    self.to_fwd:fill(0)
	    for i=1,#cond do
	      self.to_fwd[ cond[i] ] = self.to_fwd[ cond[i] ] + 1 
	      self.output[ i ]:copy( self.temp_out[ cond[i]][self.to_fwd[cond[i]]])
	    end
	 else
	    local same = true
	    local c = cond[1]
	    for i=2,#cond do if cond[i]~=c then same=false; break end end
	    if same then
	       self.output:addmm(0, self.output, 1, input, self.getWeight(c):t())
	    else
	       for i=1,nframe do
		  assert(cond[i]<=self.ncond, "tag " .. cond[i] .. " unknown")
		  self.output[i]:addmv(1, self.getWeight( cond[i] ), input[i])
	       end
	    end
	 end
      else
	 for i=1,nframe do
	    --print("forward")
	    --print("ConditionnedLinear " .. self.ncond)
	    --print(cond)
	    assert(cond[i]<=self.ncond, "tag " .. cond[i] .. " unknown")
	    self.output[i]:addmv(1, self.getWeight( cond[i] ), input[i])
	    --self.output:addmm(0, self.output, 1, input, self.weight:t())
	 end   
      end
      if self.bias then self.output:addr(1, self.addBuffer, self.bias) end
   else
      error('input must be vector or matrix')
   end
   -- print("output")
   -- print(self.output)
   return self.output
end

function ConditionedLinear:updateGradInput(inp, gradOutput)
   local input = inp[1]
   local cond = inp[2]
   local nframe = input:size(1)
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
	 self.gradInput:addmv(0, 1, self.getWeight( cond[1] ):t(), gradOutput)
      elseif input:dim() == 2 then
	 if self.optim then
	    local same = true
	    local c = cond[1]
	    for i=2,#cond do if cond[i]~=c then same=false; break end end
	    if same then
	       self.gradInput:addmm(0, 1, gradOutput, self.getWeight(c))
	    else
	       for i=1,nframe do
		  self.gradInput[i]:addmv(0, 1, self.getWeight( cond[i] ):t(), gradOutput[i])
	       end
	    end
	 else
	    for i=1,nframe do
	       self.gradInput[i]:addmv(0, 1, self.getWeight( cond[i] ):t(), gradOutput[i])
	    end
	    --self.gradInput:addmm(0, 1, gradOutput, self.weight)
	 end
      end

      return {self.gradInput, {}}
   end
end

function ConditionedLinear:accGradParameters(inp, gradOutput, scale)
   local input = inp[1]
   local cond = inp[2]
   scale = scale or 1
   if input:dim() == 1 then
      self.getGradWeight( cond[1] ):addr(scale, gradOutput, input)
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      if self.optim then
	 local same = true
	 local c = cond[1]
	 for i=2,#cond do if cond[i]~=c then same=false; break end end
	 if same then
	    self.getGradWeight(c):addmm(scale, gradOutput:t(), input)
	 else
	    for i=1,nframe do
	       self.getGradWeight( cond[i] ):addr(scale, gradOutput[i], input[i])
	    end
	 end
      else
	 for i=1,nframe do
	    self.getGradWeight( cond[i] ):addr(scale, gradOutput[i], input[i])
	 end
	 --self.gradWeight:addmm(scale, gradOutput:t(), input)
	 
      end
      if self.bias then
	 -- update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
	 updateAddBuffer(self, input)
	 self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
      end
   end
end

-- we do not need to accumulate parameters when sharing
ConditionedLinear.sharedAccUpdateGradParameters = ConditionedLinear.accUpdateGradParameters

function ConditionedLinear:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function ConditionedLinear:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
