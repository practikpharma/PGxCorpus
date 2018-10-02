local Lattice = require 'lattice'

require 'nn'

local Tagger = torch.class('nn.TagInferenceBIOES')

local function co_node_iterator(input, gradinput, stopinput, stopgradinput, labelhash, T)
   local stopinput_p = stopinput:data()
   local stopgradinput_p = stopgradinput:data()
   local input_p = input:data()
   local gradinput_p = gradinput:data()
   local nlabel = #labelhash

   for t=1,T+1 do
      if t==T+1 then
         coroutine.yield(t, 1, stopinput_p, stopgradinput_p)
      else
         for j=1,nlabel do
            coroutine.yield(t, j, input_p+(t-1)*nlabel+j-1, gradinput_p+(t-1)*nlabel+j-1)
         end
      end
   end
end

local function co_edge_iterator(trans, gradtrans, stop, gradstop, labelhash, T)
   local trans_p = trans:data()
   local stop_p = stop:data()
   local gradtrans_p = gradtrans:data()
   local gradstop_p = gradstop:data()
   local nlabel = #labelhash

   for t=2,T+1 do
      if t==T+1 then
         for jm=1,nlabel do
            local lblm = labelhash[jm]
            if lblm == 'O' or lblm:match('^E%-') or lblm:match('^S%-') then
               coroutine.yield(t, 1, t-1, jm, stop_p+jm-1, gradstop_p+jm-1)
            end
         end
      else
--         print('------------------------------------------------------------------')
         for j=1,nlabel do
            local lbl = labelhash[j]
            for jm=1,nlabel do
               local lblm = labelhash[jm]
               local tagm = lblm:gsub('^.%-', '')
               if t > 2 or lblm:match('^B%-') or lblm:match('^S%-') or lblm == 'O' then
                  if (lblm == 'O' and (lbl == 'O' or lbl:match('^B%-') or lbl:match('^S%-')))
                  or (lblm:match('^B%-') and (lbl == 'I-' .. tagm or lbl == 'E-' .. tagm))
                  or (lblm:match('^I%-') and (lbl == 'I-' .. tagm or lbl == 'E-' .. tagm))
                  or (lblm:match('^E%-') and (lbl == 'O' or lbl:match('^B%-') or lbl:match('^S%-')))
                  or (lblm:match('^S%-') and (lbl == 'O' or lbl:match('^B%-') or lbl:match('^S%-')))
                  then
--                  print(string.format("%s -> %s", lblm, lbl))
                     coroutine.yield(t, j, t-1, jm, trans_p+(j-1)*nlabel+jm-1, gradtrans_p+(j-1)*nlabel+jm-1)
                  end
               end
            end
         end
      end
   end
end

local function create_lattice(self)
   local nnode = 0
   local function f_node()
      local co = coroutine.create(co_node_iterator)
      return function()
                local flag, i, j, ip, gip = coroutine.resume(co,
                                                             self.input, self.gradinput,
                                                             self.stopinput, self.stopgradinput,
                                                             self.labelhash,
                                                             self.maxT)
                --             print(i, j, ip, gip)
                if not flag then
                   error('internal error: '..  i)
                end
                if i then
                   nnode = nnode + 1
                end
                return i, j, ip, gip
             end
   end

   local nedge = 0
   local function f_edge()
      local co = coroutine.create(co_edge_iterator)
      return function()
                local flag, i, j, im, jm, ip, gip = coroutine.resume(co,
                                                                     self.trans, self.gradtrans,
                                                                     self.stop, self.gradstop,
                                                                     self.labelhash,
                                                                     self.maxT)
                --             print(flag, i, j, im, jm, ip, gip)
                if not flag then
                   error('internal error: ' .. i)
                end
                if i then
                   nedge = nedge + 1
                end
                return i, j, im, jm, ip, gip
             end
   end

   self.lattice = Lattice.new(self.maxT+1, #self.labelhash, f_node, f_edge)
   print(string.format('lattice created with %d nodes and %d edges', nnode, nedge))
end

function Tagger:__init(labelhash, maxT)
   for i=1,#labelhash do
      assert(labelhash[i]=="O" or labelhash[i]:match("^B%-") or labelhash[i]:match("^I%-")
		or labelhash[i]:match("^E%-") or labelhash[i]:match("^S%-"), "error")
   end
   
   self.nlabel = #labelhash
   self.labelhash = labelhash
   self.maxT = maxT
   self.gradInput = {}
   self.input = torch.rand(maxT, self.nlabel)
   self.gradinput = torch.zeros(maxT, self.nlabel)
   self.trans = torch.rand(self.nlabel, self.nlabel)
   self.stop  = torch.zeros(self.nlabel)
   self.gradtrans = torch.rand(self.nlabel, self.nlabel)
   self.gradstop  = torch.zeros(self.nlabel)
   self.stopinput = torch.zeros(1)
   self.stopgradinput = torch.zeros(1)

   create_lattice(self)

   return self
end

function Tagger:forward_max(input)
   local T = input:size(1)
   assert(T <= self.maxT, 'input too long')
   self.input:narrow(1, self.maxT-T+1, T):copy(input)

   local score, path_ = self.lattice:forward(T+1) -- do not forget end node

   -- convert the path to something descent
   local path = torch.Tensor(path_.size-1) -- we do not report last node
                                           -- which is the same end node for
                                           -- everyone
   local idx = 0
   for i=path_.size-1, 1, -1 do
      idx = idx + 1
      path[idx] = path_.path[i].idx
   end
   
   return score, path
end

function Tagger:forward_logadd(input)
   local T = input:size(1)
   assert(T <= self.maxT, 'input too long')
   self.input:narrow(1, self.maxT-T+1, T):copy(input)
   local score = self.lattice:forward_logadd(T+1) -- do not forget end node
   return score
end

function Tagger:zeroGradParameters()
   self.gradtrans:zero()
   self.gradstop:zero()
end

function Tagger:zeroGradInput(input)
   local T = input:size(1)
   assert(T <= self.maxT, 'input too long')
   self.gradinput:narrow(1, self.maxT-T+1, T):zero()
end

function Tagger:backward_max(input, g)
   local T = input:size(1)
   assert(T <= self.maxT, 'input too long')
   self.input:narrow(1, self.maxT-T+1, T):copy(input)
   self.lattice:backward(g, T+1)
   self.gradInput = self.gradinput:narrow(1, self.maxT-T+1, T)
   return self.gradInput
end

function Tagger:backward_logadd(input, g)
   local T = input:size(1)
   assert(T <= self.maxT, 'input too long')
   self.input:narrow(1, self.maxT-T+1, T):copy(input)
   self.lattice:backward_logadd(g, T+1) -- WTF???
   self.gradInput = self.gradinput:narrow(1, self.maxT-T+1, T)
   return self.gradInput
end

local function printt(tbl)
   for i=1,#tbl do
      print(i, tbl[i])
   end
end

function Tagger:forward_correct(input, path)
--    local path_ = {}
--    for i=path:size(1),1,-1 do
--       table.insert(path_, (path[i][1]-1)*self.nlabel + path[i][2])
--       table.insert(path_, path[i][1])
--    end
--    printt(path_)
--    print("POUF")
--    return self.lattice:forward_correct(path_)
   local offset = self.maxT-input:size(1)
   local sum = 0
   local t = 1+offset
   local labelp
   for i=1,path:size(1) do
      local label = path[i]
      sum = sum + self.input[t][label]
      if labelp then
         sum = sum + self.trans[label][labelp]
      end
      labelp = label
      t = t + 1
   end
   return sum
end

function Tagger:backward_correct(input, path, g)
   local offset = self.maxT-input:size(1)
   local sum = 0
   local t = 1+offset
   local labelp
   for i=1,path:size(1) do
      local label = path[i]
      self.gradinput[t][label] = self.gradinput[t][label] + g
      if labelp then
         self.gradtrans[label][labelp] = self.gradtrans[label][labelp] + g
      end
      labelp = label
      t = t + 1
   end
   return sum
end

function Tagger:updateParameters(lr)
   self.trans:add(-lr, self.gradtrans)
end

function Tagger:toString(sz)
   self.lattice:toString(sz)
end

function Tagger:write(file)
   file:writeInt(self.nlabel)
   file:writeObject(self.labelhash)
   file:writeInt(self.maxT)
   file:writeObject(self.gradInput)
   file:writeObject(self.input)
   file:writeObject(self.gradinput)
   file:writeObject(self.trans)
   file:writeObject(self.stop)
   file:writeObject(self.gradtrans)
   file:writeObject(self.gradstop)
   file:writeObject(self.stopinput)
   file:writeObject(self.stopgradinput)
end

function Tagger:read(file)
   self.nlabel = file:readInt()
   self.labelhash = file:readObject()
   self.maxT = file:readInt()
   self.gradInput = file:readObject()
   self.input = file:readObject()
   self.gradinput = file:readObject()
   self.trans = file:readObject()
   self.stop = file:readObject()
   self.gradtrans = file:readObject()
   self.gradstop = file:readObject()
   self.stopinput = file:readObject()
   self.stopgradinput = file:readObject()
   create_lattice(self)
end
