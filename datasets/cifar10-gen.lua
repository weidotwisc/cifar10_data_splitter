--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This automatically downloads the CIFAR-10 dataset from
--  http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz
--

local URL = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'

local M = {}

-- added by Wei 2018-05-05 partition training data to train data and validation data
-- ratio is how much data is to use as training data, e.g., 0.95
local function convertToTensorTV(files, ratio)
   local data, labels

   for _, file in ipairs(files) do
      local m = torch.load(file, 'ascii')
      if not data then
         data = m.data:t()
         labels = m.labels:squeeze()
      else
         data = torch.cat(data, m.data:t(), 1)
         labels = torch.cat(labels, m.labels:squeeze())
      end
   end

   -- This is *very* important. The downloaded files have labels 0-9, which do
   -- not work with CrossEntropyCriterion
   labels:add(1)
  
   data = data:contiguous():view(-1, 3, 32, 32)
   labels = labels

   -- the below is Wei's logic to seperate train and val datasets 2018-05-05
   cnt = data:size()[1]
   perm = torch.randperm(cnt):long() -- when use Torch Tensor index, it requires a long a Long tensor
   trnPerm = perm:narrow(1,1,cnt*ratio)
   valPerm = perm:narrow(1,cnt*ratio+1, cnt*(1.0-ratio)) -- apparently, in Lua 1.0 is different from 1, say if i have cnt=50000, ratio=0.8, if i use 1 then this experission is 9999 instead of 10000, weird!
   torch.save('gen/cifar10-trnPerm.t7', trnPerm)
   torch.save('gen/cifar10-valPerm.t7', valPerm)
   trnData = data:index(1,trnPerm)
   trnLabels = labels:index(1,trnPerm)
   valData = data:index(1, valPerm)
   valLabels = labels:index(1,valPerm)
   return
      {
	 data = trnData,
	 labels = trnLabels,
      },
      {
	 data = valData,
	 labels = valLabels,
      }
   
   --- end of Wei's edits
end


local function convertToTensor(files)
   local data, labels

   for _, file in ipairs(files) do
      local m = torch.load(file, 'ascii')
      if not data then
         data = m.data:t()
         labels = m.labels:squeeze()
      else
         data = torch.cat(data, m.data:t(), 1)
         labels = torch.cat(labels, m.labels:squeeze())
      end
   end

   -- This is *very* important. The downloaded files have labels 0-9, which do
   -- not work with CrossEntropyCriterion
   labels:add(1)

   return {
      data = data:contiguous():view(-1, 3, 32, 32),
      labels = labels,
   }
end

local function fileDownloaded()
   return paths.filep('gen/cifar-10-torch.tar.gz')	 
end

function M.exec(opt, cacheFile)
   local hasDownloaded = fileDownloaded()
   if(not hasDownloaded) then
      print("=> Downloading CIFAR-10 dataset from " .. URL)
      local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
      assert(ok == true or ok == 0, 'error downloading CIFAR-10')
   end
   print(" | combining dataset into a single file")
   local trainData, valData = convertToTensorTV({
      'gen/cifar-10-batches-t7/data_batch_1.t7',
      'gen/cifar-10-batches-t7/data_batch_2.t7',
      'gen/cifar-10-batches-t7/data_batch_3.t7',
      'gen/cifar-10-batches-t7/data_batch_4.t7',
      'gen/cifar-10-batches-t7/data_batch_5.t7',
   }, opt.tvRatio)
   local testData = convertToTensor({
      'gen/cifar-10-batches-t7/test_batch.t7',
   })

   print(" | saving CIFAR-10 dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = valData,
      test = testData
   })
end

return M
