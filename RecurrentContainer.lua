local RecurrentContainer, parent = torch.class('srnn.RecurrentContainer', 'srnn.SequenceContainer')

function RecurrentContainer:__init(inputSize, outputSize, layers, dropout, updateInit, shareModule)
	self.nlayer = layers or 1
	self.inputSize = inputSize
	self.outputSize = outputSize or inputSize
	parent.__init(self, self:buildStep(inputSize, self.outputSize, self.nlayer, dropout), shareModule)
	self.initStateStorage = nn.Module()
	self:add(self.initStateStorage)
	self.updateInit = (updateInit == nil) and true or updateInit
	self:reset()
	self:training()
end

function RecurrentContainer:updateOutput(input)
	self:prepareForward(input)
	local _input = self:getInput(self.fwd_step, input)
	self.output = self:net(self.fwd_step):updateOutput(_input)[self.nlayer]
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
	if (self.acg_step == 1) and self.updateInit then
		for _, unit in ipairs(self:net(1).gradInput) do
			if _ > 1 then
				self.initStateStorage.gradWeight[_ - 1]:add(unit:sum(1))
			end
		end
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
	if (self.acg_step == 1) and self.updateInit then
		for _, unit in ipairs(self:net(1).gradInput) do
			if _ > 1 then
				self.initStateStorage.gradWeight[_ - 1]:add(unit:sum(1))
			end
		end
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

function RecurrentContainer:reset()
	self:resetStep()
	parent.reset(self)
end

function RecurrentContainer:clearState()
	self:resetStep()
	return parent.clearState(self)
end

function RecurrentContainer:resetStep()
	self.fwd_step = 1
	self.ugi_step = 1
	self.acg_step = 1
	self.inputs = {}
	self.gradOutputs = {}
	self.forwarded = false
	self.backwarded = true
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
