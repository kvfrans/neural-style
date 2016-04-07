require "torch"
require "nn"
require "image"
require "paths"
require "optim"
local pl = require('pl.import_into')()
local printf = pl.utils.printf

-- import the VGG convnet model-creation functions
paths.dofile('util.lua')
paths.dofile('costs.lua')
paths.dofile('image.lua')
paths.dofile('vgg.lua')

local model = create_vgg("models/vgg_normalized.th","cunn")
local style_weights = {
        ['conv1_1'] = 1,
        ['conv2_1'] = 1,
        ['conv3_1'] = 1,
        ['conv4_1'] = 1,
        ['conv5_1'] = 1,
}

local content_weights = {
        ['conv4_2'] = 1,
}
model:float()
collectgarbage()

-- compute normalization factor
local style_weight_sum = 0
local content_weight_sum = 0
for k, v in pairs(style_weights) do
    style_weight_sum = style_weight_sum + v
end

for k, v in pairs(content_weights) do
    content_weight_sum = content_weight_sum + v
end

local img = preprocess(image.load("bee.jpg"), 200)
-- print(img:size())
-- image.display(img)
model:forward(img)
local img_activations, _ = collect_activations(model, content_weights, {})

local art = preprocess(
    image.load("sky.jpg"), math.max(img:size(3), img:size(4))
)
-- print(art:size())
model:forward(art)
local _, art_grams = collect_activations(model, {}, style_weights)
art = nil
collectgarbage()

-- print(model.output:size())

function optimize(input)
    model:forward(input)
    local loss = 0
    local grad = torch.FloatTensor()
    grad:resize(model.output:size()):zero()
    for i = #model.modules,1,-1 do
        -- if i == 1 then return input, else use previous layer's output as input
        local module_input = (i == 1) and input or model.modules[i - 1].output
        local module = model.modules[i]
        local name = module._name

        -- add content gradient
        if name and content_weights[name] then
            local c_loss, c_grad = content_grad(module.output, img_activations[name])
            local w = content_weights[name] / content_weight_sum
            --printf('[content]\t%s\t%.2e\n', name, w * c_loss)
            loss = loss + w * c_loss
            grad:add(w, c_grad)
        end
        -- add style gradient
        if name and style_weights[name] then
            local s_loss, s_grad = style_grad(module.output, art_grams[name])
            local w = 2e9 * style_weights[name] / style_weight_sum
            --printf('[style]\t%s\t%.2e\n', name, w * s_loss)
            loss = loss + w * s_loss
            grad:add(w, s_grad)
        end
        grad = module:backward(module_input, grad)
    end

    -- total variation regularization for denoising
    -- grad:add(total_var_grad(input):mul(opt.smoothness))
    return loss, grad:view(-1)
end



local input = img
local timer = torch.Timer()
local output = depreprocess(input):double()
image.display(output)
local optim_state = {
        momentum = 0.9,
        dampening = 0.0,
        learningRate = 1e-3,
}

for i = 1, 500 do
    local _, loss = optim["sgd"](optimize, input, optim_state)
    loss = loss[1]
    if i % 100 == 0 then
        optim_state.learningRate = 0.75 * optim_state.learningRate
    end
    if i % 10 == 0 then
        printf('iter %5d\tloss %8.2e\tlr %8.2e\ttime %4.1f\n',
            i, loss, optim_state.learningRate, timer:time().real)
    end
end
local output = depreprocess(input):double()
image.display(output)
