local LenRecurrentContainer, parent = torch.class('srnn.LenRecurrentContainer', 'srnn.RecurrentContainer')

function LenRecurrentContainer:__init(...)
	parent.__init(self, ...)
	self.gradInput = {}
	self.farGradWeight = torch.Tensor()
end

function LenRecurrentContainer:updateGradInput(input, gradOutput)
	local lastStep = false
	if self.forwarded then
		self.ugi_step = self.fwd_step - 1
		self.acg_step = self.ugi_step
		self.forwarded = false
		self.backwarded = true
		self.fwd_step = 1
		lastStep = true
		if not self.gradInput[2] then
			self.gradInput[2] = input[2].new()
		end
		if not self.gradInput[2]:isSize(input[2]:size()) then
			self.gradInput[2]:resizeAs(input[2]):zero()
		end
	end
	local _input = self:getInput(self.ugi_step, input)
	local _gradOutput = self:getGradOutput(self.ugi_step, gradOutput, lastStep, input)
	self.gradInput[1] = self:net(self.ugi_step):updateGradInput(_input, _gradOutput)[1]
	if self.ugi_step > 1 then
		self.ugi_step = self.ugi_step - 1
	end
	return self.gradInput
end

function LenRecurrentContainer:accGradParameters(input, gradOutput, scale)
	local lastStep = false
	if self.forwarded then
		self.acg_step = self.fwd_step - 1
		self.ugi_step = self.acg_step
		self.forwarded = false
		self.backwarded = true
		self.fwd_step = 1
		lastStep = true
		if not self.gradInput[2] then
			self.gradInput[2] = input[2].new()
		end
		if not self.gradInput[2]:isSize(input[2]:size()) then
			self.gradInput[2]:resizeAs(input[2]):zero()
		end
	end
	local _input = self:getInput(self.acg_step, input)
	local _gradOutput = self:getGradOutput(self.acg_step, gradOutput, lastStep, input)
	self:net(self.acg_step):accGradParameters(_input, _gradOutput, scale)
	if (self.acg_step == 1) and ((#self.initStates > 0) or self.updateInit) then
		local _gt = self:net(1).gradInput
		for _ = 2, #_gt do
			local unit = _gt[_]
			if self.stepFull > 0 then
				unit:add(self.farGradWeight[_ - 1])
			end
			if self.initStates[_ - 1] then
				self.gradInitStates[_ - 1]:add(unit)
			elseif self.updateInit then
				self.initStateStorage.gradWeight[_ - 1]:add(unit:sum(1))
			end
		end
		self.initStates = {}
	end
	if self.acg_step > 1 then
		self.acg_step = self.acg_step - 1
	end
end

function LenRecurrentContainer:backward(input, gradOutput, scale)
	local lastStep = false
	if self.forwarded then
		self.acg_step = self.fwd_step - 1
		self.ugi_step = self.acg_step
		self.forwarded = false
		self.backwarded = true
		self.fwd_step = 1
		lastStep = true
		if not self.gradInput[2] then
			self.gradInput[2] = input[2].new()
		end
		if not self.gradInput[2]:isSize(input[2]:size()) then
			self.gradInput[2]:resizeAs(input[2]):zero()
		end
	end
	local _input = self:getInput(self.acg_step, input)
	local _gradOutput = self:getGradOutput(self.acg_step, gradOutput, lastStep, input)
	self.gradInput[1] = self:net(self.acg_step):backward(_input, _gradOutput, scale)[1]
	if (self.acg_step == 1) and ((#self.initStates > 0) or self.updateInit) then
		local _gt = self:net(1).gradInput
		for _ = 2, #_gt do
			local unit = _gt[_]
			if self.stepFull > 0 then
				unit:add(self.farGradWeight[_ - 1])
			end
			if self.initStates[_ - 1] then
				self.gradInitStates[_ - 1]:add(unit)
			elseif self.updateInit then
				self.initStateStorage.gradWeight[_ - 1]:add(unit:sum(1))
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

function LenRecurrentContainer:clearState()
	self.farGradWeight:set()
	self.gradInput = {}
	self.sind:set()
	self.ind:set()
	return parent.clearState(self)
end

function LenRecurrentContainer:resetStep(fwd, bwd, stepFull)
	self.stepFull = stepFull or 0
	self.realInitStates = {}
	parent.resetStep(self, fwd, bwd)
end
