local AbstractRecurrent, parent = torch.class('srnn.AbstractRecurrent', 'srnn.RecurrentContainer')

function AbstractRecurrent:prepareForward(input)
	local function reset_Table(tableIn, bsize, osize)
		for _, unit in ipairs(tableIn) do
			unit:resize(bsize, osize):zero()
		end
	end
	-- if it is the first step, reset gradients for this process
	if self.train and self.backwarded then
		local bsize = input:size(1)
		reset_Table(self.gradOutputLast, bsize, self.outputSize)
		self:resetStep(true, false)
	end
end

function AbstractRecurrent:reset()
	local function prepareTensorTable(nTensor)
		local rs = {}
		for _ = 1, nTensor do
			table.insert(rs, torch.Tensor())
		end
		return rs
	end
	if not self.gradOutputLast then
		self.gradOutputLast = prepareTensorTable(self.nlayer)
	else
		for _, unit in ipairs(self.gradOutputLast) do
			unit:set()
		end
	end
	if self.initStateStorage.weight then
		self.initStateStorage.weight:resize(self.nlayer, self.outputSize):zero()
		self.initStateStorage.gradWeight:resize(self.nlayer, self.outputSize):zero()
	else
		self.initStateStorage.weight = torch.zeros(self.nlayer, self.outputSize)
		self.initStateStorage.gradWeight = torch.zeros(self.nlayer, self.outputSize)
	end
	parent.reset(self)
end

function AbstractRecurrent:getInput(step, input)
	if self.inputs[step] then
		return self.inputs[step]
	else
		local _input = {input}
		-- for the first step, use init outputs and cells, for the later, use the output of the previous step module
		if step > 1 then
			for _, unit in ipairs(self:net(step - 1).output) do
				table.insert(_input, unit)
			end
		else
			local batchsize = input:size(1)
			for _ = 1, self.nlayer do
				table.insert(_input, self.initStateStorage.weight[_]:reshape(1, self.outputSize):expand(batchsize, self.outputSize))
			end
		end
		if self.train then
			self.inputs[step] = _input
		end
		return _input
	end
end

function AbstractRecurrent:getGradOutput(step, gradOutput, lastStep)
	local function getCore(step, gradOutput, lastStep)
		local function buildTable(tba, tbta)
			for _, unit in ipairs(tbta) do
				table.insert(tba, unit)
			end
		end
		if self.gradOutputs[step] then
			return self.gradOutputs[step]
		else
			local _gradOutput = {}
			-- if this is the first time to backward, set step to current step and build _gradOutput with self.gradOutputLast and self.gradCellLast, otherwise build with the gradInput of the next step module's gradInput
			if lastStep then
				buildTable(_gradOutput, self.gradOutputLast)
			else
				local _gt = self:net(step + 1).gradInput
				for _, unit in ipairs(_gt) do
					if _ > 1 then
						table.insert(_gradOutput, unit)
					end
				end
				-- assume that updateGradInput was called at first, while accGradParameters, gradOutput will not be added for a second time
				if gradOutput then
					_gradOutput[self.nlayer]:add(gradOutput)
				end
				-- apply extra gradient if there is any
				if self.gradOutputAdd[step] then
					for _, grad in ipairs(self.gradOutputAdd[step]) do
						_gradOutput[_]:add(grad)
					end
					self.gradOutputAdd[step] = nil
				end
			end
			self.gradOutputs[step] = _gradOutput
			return _gradOutput
		end
	end
	local rs = getCore(step, gradOutput, lastStep)
	-- for only one layer, just retrieve the only gradOutput
	if #rs == 1 then
		return rs[1]
	else
		return rs
	end
end

function AbstractRecurrent:getStepOutput(step, layer)
	return self:net(step).output[layer or self.nlayer]
end

function AbstractRecurrent:setStepGradOutputAdd(gradToAdd, step, layer)
	if not self.gradOutputAdd[step] then
		self.gradOutputAdd[step] = {}
	end
	if self.gradOutputAdd[step][layer or self.nlayer] then
		self.gradOutputAdd[step][layer or self.nlayer]:add(gradToAdd)
	else
		self.gradOutputAdd[step][layer or self.nlayer] = gradToAdd
	end
end

function AbstractRecurrent:resetStep(...)
	self.gradOutputAdd = {}
	parent.resetStep(self, ...)
end
