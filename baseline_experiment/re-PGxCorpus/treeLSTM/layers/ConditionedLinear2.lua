local ConditionedLinear2, parent = torch.class('nn.ConditionedLinear2', 'nn.Module')

function ConditionedLinear2:__init(inputSize, outputSize, bias, ncond, optim)
   parent.__init(self)
   self.ncond = ncond or error("ncond must be entered")  
   local bias = ((bias == nil) and true) or bias
   self.weight = torch.Tensor(outputSize, inputSize * self.ncond * self.ncond)
   self.gradWeight = torch.Tensor(outputSize, inputSize * self.ncond * self.ncond)
   self.weights = {}
   self.gradWeights = {}
   self.optim = optim
      
   self.getWeight = function(i,j) 
      local k = ((i-1)*self.ncond) + (j-1)
      return self.weight:narrow(2,(k*inputSize)+1, inputSize)
   end
   self.getGradWeight = function(i,j)
      local k = ((i-1)*self.ncond) + (j-1)
      return self.gradWeight:narrow(2,(k*inputSize)+1, inputSize)
   end
   
   -- for i=1,self.ncond do
   --    self.weights[i] = {}
   --    self.gradWeights[i] = {}
   --    for j=1,self.ncond do
   -- 	 local k = ((i-1)*self.ncond) + (j-1)
   -- 	 self.weights[i][j] = 
   -- 	 self.gradWeights[i][j] = 
   --    end
   -- end
   
   if bias then
      self.bias = torch.Tensor(outputSize)
      self.gradBias = torch.Tensor(outputSize)
   end
 
   self:reset()
end

function ConditionedLinear2:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function ConditionedLinear2:reset(stdv)
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

function ConditionedLinear2:updateOutput(inp)
   local input = inp[1]
   local cond = inp[2]
   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.bias then self.output:copy(self.bias) else self.output:zero() end
      self.output:addmv(1, self.getWeight(cond[1],cond[1]), input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, nframe, self.weight:size(1))
      self.output:fill(0)
      updateAddBuffer(self, input)
      local same
      if self.optim then
	 same = true
	 local c = cond[1]
	 for i=2,#cond do if cond[i]~=c then same=false; break end end
      end
      if self.optim and same then
	 self.output:resize(nframe, self.weight:size(1)):fill(0)
	 self.output:addmm(0, self.output, 1, input, self.getWeight(cond[1], cond[1]):t())
	 if self.bias then self.output:addr(1, self.addBuffer, self.bias) end

	 self.output:resize(1, self.output:size(1), self.output:size(2))
	 self.output:expand(self.output, nframe, nframe, self.weight:size(1))
      else
	 for i=1,nframe do
	    for j=1,nframe do
	       assert(cond[i]<=self.ncond
			 and cond[j]<=self.ncond, "tag " .. cond[i] .. " or " .. cond[j] .. " unknown")
	       self.output[i][j]:addmv(1, self.getWeight(cond[i],cond[j]), input[j])
	       --self.output:addmm(0, self.output, 1, input, self.weight:t())
	    end
	 end
	 for i=1, nframe do
	    if self.bias then self.output[i]:addr(1, self.addBuffer, self.bias) end
	 end
      end
   else
      error('input must be vector or matrix')
   end
   return self.output
end

function ConditionedLinear2:updateGradInput(inp, gradOutput)
   local input = inp[1]
   local cond = inp[2]
   if self.gradInput then
      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
	 self.gradInput:addmv(0, 1, self.getWeight(cond[1],cond[1]):t(), gradOutput)
      elseif input:dim() == 2 then
	 local nframe = input:size(1)
	 local same
	 if self.optim then
	    same = true
	    local c = cond[1]
	    for i=2,#cond do if cond[i]~=c then same=false; break end end
	 end
	 if self.optim and same then
	    --print("optimized version")
	    --caution: sum allocate memory. Fix this
	    self.gradInput:addmm(0, 1, gradOutput:sum(1):resize(gradOutput:size(2), gradOutput:size(3)), self.getWeight(cond[1], cond[1]))
	 else
	    --print("not optimized version")
	    self.gradInput:fill(0)
	    for i=1,nframe do
	       for j=1,nframe do
		  self.gradInput[j]:addmv(1, 1, self.getWeight(cond[i],cond[j]):t(), gradOutput[i][j])
	       end
	    end
	    --self.gradInput:addmm(0, 1, gradOutput, self.weight)
	 end
      end

      return {self.gradInput, {}}
   end
end

function ConditionedLinear2:accGradParameters(inp, gradOutput, scale)
   local input = inp[1]
   local cond = inp[2]
   scale = scale or 1
   if input:dim() == 1 then
      self.getGradWeight(cond[1],cond[1]):addr(scale, gradOutput, input)
      if self.bias then self.gradBias:add(scale, gradOutput) end
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local same
      if self.optim then
	 same = true
	 local c = cond[1]
	 for i=2,#cond do if cond[i]~=c then same=false; break end end
      end
      if self.optim and same then
	 local temp = gradOutput:sum(1):resize(gradOutput:size(2), gradOutput:size(3)):t()--caution: allocation
	 if self.bias then
	    self.gradBias:addmv(scale, temp, self.addBuffer)
	 end
	 self.getGradWeight(cond[1],cond[1]):addmm(scale, temp, input)
      else
	 for i=1,nframe do
	    for j=1,nframe do
	       self.getGradWeight(cond[i],cond[j]):addr(scale, gradOutput[i][j], input[j])
	    end
	 end
	 --self.gradWeight:addmm(scale, gradOutput:t(), input)
	 if self.bias then
	    -- update the size of addBuffer if the input is not the same size as the one we had in last updateGradInput
	    updateAddBuffer(self, input)
	    for i=1,nframe do
	       self.gradBias:addmv(scale, gradOutput[i]:t(), self.addBuffer)
	    end
	 end
      end
   end
end

-- we do not need to accumulate parameters when sharing
ConditionedLinear2.sharedAccUpdateGradParameters = ConditionedLinear2.accUpdateGradParameters

function ConditionedLinear2:clearState()
   if self.addBuffer then self.addBuffer:set() end
   return parent.clearState(self)
end

function ConditionedLinear2:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1)) ..
      (self.bias == nil and ' without bias' or '')
end
