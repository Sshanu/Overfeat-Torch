require 'nn'
require 'image'
label     = require 'overfeat_label'
torch.setdefaulttensortype('torch.FloatTensor')

-- Model
net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 96, 11, 11, 4, 4))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.SpatialConvolution(96, 256, 5, 5, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
net:add(nn.SpatialConvolution(1024, 3072, 6, 6, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(3072, 4096, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(4096, 1000, 1, 1, 1, 1))
net:add(nn.View(1000))
net:add(nn.SpatialSoftMax())
net=torch.load('model.net')

-- load and preprocess image
print('==> prepare an input image')
img = image.load('input_img.jpg'):mul(255)
 dim = 231 or 221
img_scale = image.scale(img,dim,dim)
img=img_scale

-- feedforward network
print('==> feed the input image')
timer = torch.Timer()
img:add(-118.380948):div(61.896913)
local out = net:forward(img)
local topN = 10
local probs, idxs = torch.topk(out, topN, 1, true)
print('==> Results')
for i=1,topN do
    print(i..":"..label[idxs[i]].." =", probs[i])
end

print('')
print('==> Top result')
local prob, idx = torch.max(out, 1)

print(label[idx:squeeze()], prob:squeeze())

print('Time elapsed: ' .. timer:time().real .. ' seconds')
