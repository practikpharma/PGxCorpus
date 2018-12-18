require "nn"
dofile("../layers/ConditionedLinear2.lua")

torch.manualSeed(1)

--parameters
local inputsize = 100
local outputsize = 100
local nconditions = 1
local cond = {1}

local clinear = nn.ConditionedLinear2(inputsize, outputsize, true, nconditions)

local input = torch.rand(inputsize)
local target = torch.rand(outputsize)
local criterion = nn.MSECriterion()

local output = clinear:forward({input,cond})

local cost = criterion:forward(output, target)

local epsilon = 0.00001

if false then
print("***************************** vector input ***************************")
print("============================= weight =============================")
--finite gradient method (weight)
for i=1, clinear.weight:size(1) do
   for j=1, clinear.weight:size(2) do

      local back = clinear.weight[i][j]
      clinear.weight[i][j] = clinear.weight[i][j] + epsilon

      local output = clinear:forward({input, cond})
      local cost2 = criterion:forward(output, target)
      local grad = criterion:backward(output, target)
      
      local deriv = (cost2 - cost) / epsilon

      clinear:zeroGradParameters()
      local gradinput = clinear:backward({input,cond}, grad)
      
      --print(cost .. " " .. cost2)
      --print(deriv .. " " .. clinear.gradWeight[i][j])
      print(deriv - clinear.gradWeight[i][j])
      assert(math.abs(deriv - clinear.gradWeight[i][j]) < 0.0001)

      clinear.weight[i][j] = back
   end
end
print("============================= bias =============================")
for i=1, clinear.bias:size(1) do
   local back = clinear.bias[i]
   clinear.bias[i] = clinear.bias[i] + epsilon
   
   local output = clinear:forward({input, cond})
   local cost2 = criterion:forward(output, target)
   local grad = criterion:backward(output, target)
      
   local deriv = (cost2 - cost) / epsilon
   
   clinear:zeroGradParameters()
   local gradinput = clinear:backward({input,cond}, grad)
   
   --print(cost .. " " .. cost2)
   --print(deriv .. " " .. clinear.gradBias[i])
   print(deriv - clinear.gradBias[i])
   assert(math.abs(deriv - clinear.gradBias[i]) < 0.0001)
   
   clinear.bias[i] = back
end

print("============================= input =============================")
--finite gradient method (input)
for i=1, input:size(1) do
   --for j=1, clinear.weight:size(2) do
   local back = input[i]--[j]
   input[i] = input[i] + epsilon

   local output = clinear:forward({input, cond})
   local cost2 = criterion:forward(output, target)
   local grad = criterion:backward(output, target)
   
   local deriv = (cost2 - cost) / epsilon

   clinear:zeroGradParameters()
   local gradinput = clinear:backward({input,cond}, grad)
   
   --print(cost .. " " .. cost2)
   --print(deriv .. " " .. gradinput[i])
   print(deriv - gradinput[i])
   assert(math.abs(deriv - gradinput[i]) < 0.0001)
   
   input[i] = back
end
end

local ncomp = 40
local input = torch.rand(ncomp,inputsize)
local temp = input[1]:clone()
local cond = {}
for i=1,ncomp do table.insert(cond, math.random(nconditions)) end
print(cond)
local target = torch.rand(ncomp,ncomp, outputsize)
local criterion = nn.MSECriterion()

local output = clinear:forward({input,cond})
local cost = criterion:forward(output, target)

local epsilon = 0.000001

local timer = torch.Timer()
local nforward = 0
print("***************************** matrix input ***************************")
print("============================= weight =============================")
--finite gradient method (weight)
for i=1, clinear.weight:size(1) do
   for j=1, clinear.weight:size(2) do
      nforward = nforward + 1

      local back = clinear.weight[i][j]
      clinear.weight[i][j] = clinear.weight[i][j] + epsilon

      local output = clinear:forward({input, cond})
      local cost2 = criterion:forward(output, target)
      local grad = criterion:backward(output, target)
     
      local deriv = (cost2 - cost) / epsilon

      clinear:zeroGradParameters()
      local gradinput = clinear:backward({input,cond}, grad)
      
      --print(cost .. " " .. cost2)
      --print(deriv .. " " .. clinear.gradWeight[i][j])
      --print(deriv - clinear.gradWeight[i][j])
      assert(math.abs(deriv - clinear.gradWeight[i][j]) < 0.00001)

      clinear.weight[i][j] = back
      --io.read()
   end
end
print("============================= bias =============================")
for i=1, clinear.bias:size(1) do
   nforward = nforward + 1

   local back = clinear.bias[i]
   clinear.bias[i] = clinear.bias[i] + epsilon
   
   local output = clinear:forward({input, cond})
   local cost2 = criterion:forward(output, target)
   local grad = criterion:backward(output, target)
      
   local deriv = (cost2 - cost) / epsilon
   
   clinear:zeroGradParameters()
   local gradinput = clinear:backward({input,cond}, grad)
   
   --print(cost .. " " .. cost2)
   --print(deriv .. " " .. clinear.gradBias[i])
   --print(deriv - clinear.gradBias[i])
   assert(math.abs(deriv - clinear.gradBias[i]) < 0.00001)
   
   clinear.bias[i] = back
end

print("============================= input =============================")
--finite gradient method (input)
for i=1, input:size(1) do
   for j=1, input:size(2) do
      nforward = nforward + 1

      --print(i .. " " .. j)
      local back = input[i][j]
      input[i][j] = input[i][j] + epsilon
      
      local output = clinear:forward({input, cond})
      local cost2 = criterion:forward(output, target)
      local grad = criterion:backward(output, target)
      
      local deriv = (cost2 - cost) / epsilon
      
      clinear:zeroGradParameters()
      local gradinput = clinear:backward({input,cond}, grad)
      
      --print(cost .. " " .. cost2)
      --print(deriv .. " " .. gradinput[i][j])
      --print(deriv - gradinput[i][j])
      assert(math.abs(deriv - gradinput[i][j]) < 0.00001)
      
      input[i][j] = back
   end
end
   
print("Time " .. nforward/timer:time().real .. " ex/s")
