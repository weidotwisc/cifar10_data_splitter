trnP=torch.load('gen/cifar10-trnPerm.t7')
valP=torch.load('gen/cifar10-valPerm.t7')

newD = torch.load('gen/cifar10.t7')
origD = torch.load('gen/backup/cifar10.t7')
print(string.format("trnP size %d, valP size %d", trnP:size()[1], valP:size()[1]))
assert(trnP:size()[1]+valP:size()[1] == 50000)
print("TEST 0 passed " .. " didnt miss data")

-- test case 1, make sure test data are the same
t1=newD['test']['data']
t2=origD['val']['data']
assert(torch.all(torch.eq(t1,t2)), "test data are not same")
t1=newD['test']['labels']
t2=origD['val']['labels']
assert(torch.all(torch.eq(t1,t2)), "test labels are not same")
print("TEST1 passed! " .. " test datasets are the same")

-- test case2, make sure train data are correct
for i=1, trnP:size()[1] do
   idx=trnP[i]
   t1=newD['train']['data'][i]
   t2=origD['train']['data'][idx]
   assert(torch.all(torch.eq(t1,t2)), "train data are not same")
   t1=newD['train']['labels'][i]
   t2=origD['train']['labels'][idx]
   assert(t1==t2, "train labels are not same")
   
end
print("TEST 2 passed!" .. "train datasets are created correctly")


-- test case2, make sure train data are correct
for i=1, valP:size()[1] do
   idx=valP[i]
   t1=newD['val']['data'][i]
   t2=origD['train']['data'][idx]
   assert(torch.all(torch.eq(t1,t2)), "val  data are not same")
   t1=newD['val']['labels'][i]
   t2=origD['train']['labels'][idx]
   assert(t1==t2, "val labels are not same")
   
end
print("TEST 3 passed!" .. "validation datasets are created correctly")


