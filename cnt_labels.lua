function parseArgs(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('count labels in the dataset')
   cmd:text('Options:')
   cmd:option('-data', 'cifar10.t7', 'data file that we need to examine')
   cmd:option('-split', 'train', 'split to count')
   local opt = cmd:parse(arg or {})
   return opt
end

opt = parseArgs(arg)
x = torch.load(opt.data)
labs = x[opt.split].labels
total = labs:size()[1]
for i = 1, 10
do
   cnt = 0
   for l = 1, labs:size()[1] 
   do
      if(labs[l] == i) then
	 cnt = cnt + 1
      end
   end
   print(string.format("label %d has %d out of %d with a ratio %.4f", i, cnt, total, (cnt/total)))
end
