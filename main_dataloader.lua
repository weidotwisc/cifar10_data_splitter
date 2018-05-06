--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'


local opts = require 'opts'


-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)



-- Data loading
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

cnt=0
for n, sample in trainLoader:run() do
   --print(n)
   --print(sample)
   cnt = cnt +1
end
print("train loader has batches: " .. cnt)

cnt=0
for n, sample in valLoader:run() do
   --print(n)
   --print(sample)
   cnt = cnt + 1
end
print("val loader has batches: " .. cnt)

cnt = 0
for n, sample in testLoader:run() do
   --print(n)
   --print(sample)
   cnt = cnt +1
end
print("test loader has batches: " .. cnt)
