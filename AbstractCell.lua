local AbstractCell, parent = torch.class('srnn.AbstractCell', 'srnn.RecurrentContainer')

function AbstractCell:__init(...)
	local function prepareTensorTable(nTensor)
		local rs = {}
		for _ = 1, nTensor do
			table.insert(rs, torch.Tensor())
		end
		return rs
	end
	parent.__init(self, ...)
	if self.initStateStorage.weight then
		self.initStateStorage.weight:resize(self.nlayer * 2, self.outputSize):zero()
		self.initStateStorage.gradWeight:resize(self.nlayer * 2, self.outputSize):zero()
	else
		self.initStateStorage.weight = torch.zeros(self.nlayer * 2, self.outputSize)
		self.initStateStorage.gradWeight = torch.zeros(self.nlayer * 2, self.outputSize)
	end
	self.gradOutputLast = prepareTensorTable(self.nlayer)
	self.gradCellLast = prepareTensorTable(self.nlayer)
end

function AbstractCell:prepareForward(input)
	local function reset_Table(tableIn, bsize, osize)
		for _, unit in ipairs(tableIn) do
			unit:resize(bsize, osize):zero()
		end
	end
	local function pair_reset_Table(tableStd, tableSet, clear)
		for _, unit in pairs(tableStd) do
			if not tableSet[_] then
				tableSet[_] = unit.new()
			end
			tableSet[_]:resizeAs(unit):zero()
		end
		if clear then
			for _, unit in pairs(tableSet) do
				if not tableStd[_] then
					tableSet[_] = nil
				end
			end
		end
	end
	-- if it is the first step, reset gradients for this process
	if self.train and self.backwarded then
		local bsize = input:size(1)
		reset_Table(self.gradOutputLast, bsize, self.outputSize)
		reset_Table(self.gradCellLast, bsize, self.outputSize)
		if #self.initStates > 1 then
			pair_reset_Table(self.initStates, self.gradInitStates, true)
		end
		self:resetStep(true, false)
	end
end

function AbstractCell:clearState()
	local function resetTable(tbin)
		for _, unit in ipairs(tbin) do
			unit:set()
		end
	end
	resetTable(self.gradOutputLast)
	resetTable(self.gradCellLast)
	return parent.clearState(self)
end

function AbstractCell:getInput(step, input)
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
				table.insert(_input, self.initStates[_] or self.initStateStorage.weight[_]:reshape(1, self.outputSize):expand(batchsize, self.outputSize))
			end
		end
		if self.train then
			self.inputs[step] = _input
		end
		return _input
	end
end

function AbstractCell:getGradOutput(step, gradOutput, lastStep)
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
			for _ = 2, #_gt do
				table.insert(_gradOutput, _gt[_])
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

function AbstractCell:getStepCell(step, layer)
	return self:net(step).output[self.nlayer + (layer or self.nlayer)]
end

function AbstractCell:setStepGradCellAdd(gradToAdd, step, layer)
	if not self.gradOutputAdd[step] then
		self.gradOutputAdd[step] = {}
	end
	if self.gradOutputAdd[step][self.nlayer + (layer or self.nlayer)] then
		self.gradOutputAdd[step][self.nlayer + (layer or self.nlayer)]:add(gradToAdd)
	else
		self.gradOutputAdd[step][self.nlayer + (layer or self.nlayer)] = gradToAdd
	end
end

function AbstractCell:setLayerInitCellState(statein, layer)
	self.initStates[self.nlayer + (layer or self.nlayer)] = statein
end

function AbstractCell:getLayerGradInitCellState(layer)
	return self.gradInitStates[self.nlayer + (layer or self.nlayer)]
end
