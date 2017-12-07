local RecurrentContainer, parent = torch.class('srnn.RecurrentContainer', 'srnn.SequenceContainer')

function RecurrentContainer:__init(inputSize, outputSize, layers, dropout, updateInit, shareModule)
	self.nlayer = layers or 1
	self.inputSize = inputSize
	self.outputSize = outputSize or inputSize
	parent.__init(self, self:buildStep(inputSize, self.outputSize, self.nlayer, dropout), shareModule)
	self.initStateStorage = nn.Module()
	self:add(self.initStateStorage)
	self.updateInit = (updateInit == nil) and true or updateInit
	self.fwd_step = 1
	self.ugi_step = 1
	self.acg_step = 1
	self.inputs = {}
	self.gradOutputs = {}
	self.gradOutputAdd = {}
	self.initStates = {}
	self.gradInitStates = {}
	self.forwarded = false
	self.backwarded = true
	self:training()
end

function RecurrentContainer:updateOutput(input)
	self:prepareForward(input)
	local _input = self:getInput(self.fwd_step, input)
	local real_output = self:net(self.fwd_step):updateOutput(_input)
	if torch.isTensor(real_output) then
		self.output = real_output
	else
		self.output = real_output[self.nlayer]
	end
	self.fwd_step = self.fwd_step + 1
	self.forwarded = true
	return self.output
end

function RecurrentContainer:updateGradInput(input, gradOutput)
	local lastStep = false
	if self.forwarded then
		self.ugi_step = self.fwd_step - 1
		self.acg_step = self.ugi_step
		self.forwarded = false
		self.backwarded = true
		self.fwd_step = 1
		lastStep = true
	end
	local _input = self:getInput(self.ugi_step, input)
	local _gradOutput = self:getGradOutput(self.ugi_step, gradOutput, lastStep)
	self.gradInput = self:net(self.ugi_step):updateGradInput(_input, _gradOutput)[1]
	if self.ugi_step > 1 then
		self.ugi_step = self.ugi_step - 1
	end
	return self.gradInput
end

function RecurrentContainer:accGradParameters(input, gradOutput, scale)
	local lastStep = false
	if self.forwarded then
		self.acg_step = self.fwd_step - 1
		self.ugi_step = self.acg_step
		self.forwarded = false
		self.backwarded = true
		self.fwd_step = 1
		lastStep = true
	end
	local _input = self:getInput(self.acg_step, input)
	local _gradOutput = self:getGradOutput(self.acg_step, gradOutput, lastStep)
	self:net(self.acg_step):accGradParameters(_input, _gradOutput, scale)
	if (self.acg_step == 1) and ((#self.initStates > 0) or self.updateInit) then
		for _, unit in ipairs(self:net(1).gradInput) do
			if _ > 1 then
				if self.initStates[_ - 1] then
					self.gradInitStates[_ - 1]:add(unit)
				elseif self.updateInit then
					self.initStateStorage.gradWeight[_ - 1]:add(unit:sum(1))
				end
			end
		end
		self.initStates = {}
	end
	if self.acg_step > 1 then
		self.acg_step = self.acg_step - 1
	end
end

function RecurrentContainer:backward(input, gradOutput, scale)
	local lastStep = false
	if self.forwarded then
		self.acg_step = self.fwd_step - 1
		self.ugi_step = self.acg_step
		self.forwarded = false
		self.backwarded = true
		self.fwd_step = 1
		lastStep = true
	end
	local _input = self:getInput(self.acg_step, input)
	local _gradOutput = self:getGradOutput(self.acg_step, gradOutput, lastStep)
	self.gradInput = self:net(self.acg_step):backward(_input, _gradOutput, scale)[1]
	if (self.acg_step == 1) and ((#self.initStates > 0) or self.updateInit) then
		for _, unit in ipairs(self:net(1).gradInput) do
			if _ > 1 then
				if self.initStates[_ - 1] then
					self.gradInitStates[_ - 1]:add(unit)
				elseif self.updateInit then
					self.initStateStorage.gradWeight[_ - 1]:add(unit:sum(1))
				end
			end
		end
		self.initStates = {}
	end
	if self.acg_step > 1 then
		self.acg_step = self.acg_step - 1
		self.ugi_step = self.ugi_step - 1
	end
	return self.gradInput
end

function RecurrentContainer:training()
	self:resetStep()
	parent.training(self)
end

function RecurrentContainer:evaluate()
	self:resetStep()
	parent.evaluate(self)
end

function RecurrentContainer:clearState()
	self:resetStep(nil, nil, true)
	return parent.clearState(self)
end

function RecurrentContainer:resetStep(fwd, bwd)
	self.fwd_step = 1
	self.ugi_step = 1
	self.acg_step = 1
	self.inputs = {}
	self.gradOutputs = {}
	self.gradOutputAdd = {}
	if not self.train then
		self.initStates = {}
	end
	self.gradInitStates = {}
	self.forwarded = (fwd == nil) and false or fwd
	self.backwarded = (bwd == nil) and true or bwd
end

function RecurrentContainer:resetFWDStep()
	self.fwd_step = 1
end

function RecurrentContainer:setFWDStep(step)
	self.fwd_step = step
end

function RecurrentContainer:resetUGIStep()
	self.ugi_step = 1
end

function RecurrentContainer:setUGIStep(step)
	self.ugi_step = step
end

function RecurrentContainer:resetACGStep()
	self.acg_step = 1
end

function RecurrentContainer:setACGStep(step)
	self.acg_step = step
end

function RecurrentContainer:getStepOutput(step, layer)
	return self:net(step).output[layer or self.nlayer]
end

function RecurrentContainer:setStepGradOutputAdd(gradToAdd, step, layer)
	if not self.gradOutputAdd[step] then
		self.gradOutputAdd[step] = {}
	end
	if self.gradOutputAdd[step][layer or self.nlayer] then
		self.gradOutputAdd[step][layer or self.nlayer]:add(gradToAdd)
	else
		self.gradOutputAdd[step][layer or self.nlayer] = gradToAdd
	end
end

function RecurrentContainer:setInitStates(statein)
	self.initStates = statein
end

function RecurrentContainer:getGradInitStates()
	return self.gradInitStates
end

function RecurrentContainer:setLayerInitState(statein, layer)
	self.initStates[layer or self.nlayer] = statein
end

function RecurrentContainer:getLayerGradInitState(layer)
	return self.gradInitStates[layer or self.nlayer]
end
