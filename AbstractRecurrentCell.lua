local AbstractRecurrentCell, parent = torch.class('srnn.AbstractRecurrentCell', 'srnn.RecurrentContainer')

function AbstractRecurrentCell:prepareForward(input)
	local function reset_Table(tableIn, bsize, osize)
		for _, unit in ipairs(tableIn) do
			unit:resize(bsize, osize):zero()
		end
	end
	-- if it is the first step, reset gradients for this process
	if self.train and self.backwarded then
		local bsize = input:size(1)
		reset_Table(self.gradOutputLast, bsize, self.outputSize)
		reset_Table(self.gradCellLast, bsize, self.outputSize)
		self.gradOutputs = {}
		self.inputs = {}
		self.backwarded = false
	end
end

function AbstractRecurrentCell:reset()
	local function prepareTensorTable(nTensor)
		local rs = {}
		for _ = 1, nTensor do
			table.insert(rs, torch.Tensor())
		end
		return rs
	end
	if self.gradOutputLast then
		for _, unit in ipairs(self.gradOutputLast) do
			unit:set()
		end
	else
		self.gradOutputLast = prepareTensorTable(self.nlayer)
	end
	if self.gradCellLast then
		for _, unit in ipairs(self.gradCellLast) do
			unit:set()
		end
	else
		self.gradCellLast = prepareTensorTable(self.nlayer)
	end
	if self.initStateStorage.weight then
		self.initStateStorage.weight:resize(self.nlayer * 2, self.outputSize):zero()
		self.initStateStorage.gradWeight:resize(self.nlayer * 2, self.outputSize):zero()
	else
		self.initStateStorage.weight = torch.zeros(self.nlayer * 2, self.outputSize)
		self.initStateStorage.gradWeight = torch.zeros(self.nlayer * 2, self.outputSize)
	end
	parent.reset(self)
end

function AbstractRecurrentCell:getInput(step, input)
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
			for _ = 1, self.nlayer * 2 do
				table.insert(_input, self.initStateStorage.weight[_]:reshape(1, self.outputSize):expand(batchsize, self.outputSize))
			end
		end
		if self.train then
			self.inputs[step] = _input
		end
		return _input
	end
end

function AbstractRecurrentCell:getGradOutput(step, gradOutput, lastStep)
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
			buildTable(_gradOutput, self.gradCellLast)
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
		end
		self.gradOutputs[step] = _gradOutput
		return _gradOutput
	end
end
