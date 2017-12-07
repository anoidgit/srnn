local AbstractLenRecurrent, parent = torch.class('srnn.AbstractLenRecurrent', 'srnn.LenRecurrentContainer')

function AbstractLenRecurrent:__init(...)
	local function prepareTensorTable(nTensor)
		local rs = {}
		for _ = 1, nTensor do
			table.insert(rs, torch.Tensor())
		end
		return rs
	end
	parent.__init(self, ...)
	if self.initStateStorage.weight then
		self.initStateStorage.weight:resize(self.nlayer, self.outputSize):zero()
		self.initStateStorage.gradWeight:resize(self.nlayer, self.outputSize):zero()
	else
		self.initStateStorage.weight = torch.zeros(self.nlayer, self.outputSize)
		self.initStateStorage.gradWeight = torch.zeros(self.nlayer, self.outputSize)
	end
	self.gradOutputLast = prepareTensorTable(self.nlayer)
end


function AbstractLenRecurrent:prepareForward(input)
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
	if self.backwarded then
		-- prepare sind and ind
		if input[1]:type():find("torch%.Cuda.*Tensor") then
			if (not self.sind) or self.sind:type() ~= "torch.CudaLongTensor" then
				self.sind = torch.CudaLongTensor()
				self.ind = torch.CudaLongTensor()
			end
		else
			if (not self.sind) or self.sind:type() ~= "torch.LongTensor" then
				self.sind = torch.LongTensor()
				self.ind = torch.LongTensor()
			end
		end
		if self.train then
			local bsize = input[1]:size(1)
			self.farGradWeight:resize(self.nlayer, bsize, self.outputSize):zero()
			reset_Table(self.gradOutputLast, bsize, self.outputSize)
			if #self.initStates > 1 then
				pair_reset_Table(self.initStates, self.gradInitStates, true)
			end
			-- the input of the next step after padding should be the initial state,
			-- and the gradOutput of the padding steps should be cleared,
			-- Warning: here assign the first unpadding step as self.stepFull
			self:resetStep(true, false, input[2]:max() + 1)
		end
	end
end

function AbstractLenRecurrent:clearState()
	local function resetTable(tbin)
		for _, unit in ipairs(tbin) do
			unit:set()
		end
	end
	resetTable(self.gradOutputLast)
	parent.clearState(self)
end

function AbstractLenRecurrent:getInput(step, input)
	local function stepFill(bsteps, initsteps, lenvec, curstep, sind, ind, bsize, vsize)
		local mask = lenvec:ge(curstep - 1)
		local bsize = bsize or bsteps[1]:size(1)
		local vsize = vsize or initsteps[1]:size(2)
		ind:range(1, bsize)
		sind:maskedSelect(ind, mask)
		for _ = 2, #bsteps do
			bsteps[_]:indexCopy(1, sind, initsteps[_ - 1]:index(1, sind))
		end
	end
	if self.inputs[step] then
		return self.inputs[step]
	else
		local _input = {input[1]}
		-- for the first step, use init outputs and cells, for the later, use the output of the previous step module
		if step > 1 then
			for _, unit in ipairs(self:net(step - 1).output) do
				table.insert(_input, unit)
			end
			if step <= self.stepFull then
				stepFill(_input, self.realInitStates, input[2], step, self.sind, self.ind)
			end
		else
			local batchsize = input[1]:size(1)
			for _ = 1, self.nlayer do
				local _state = self.initStates[_] or self.initStateStorage.weight[_]:reshape(1, self.outputSize):expand(batchsize, self.outputSize)
				table.insert(_input, _state)
				table.insert(self.realInitStates, _state)
			end
		end
		if self.train then
			self.inputs[step] = _input
		end
		return _input
	end
end

function AbstractLenRecurrent:getGradOutput(step, gradOutput, lastStep, input)
	local function getCore(step, gradOutput, lastStep)
		local function buildTable(tba, tbta)
			for _, unit in ipairs(tbta) do
				table.insert(tba, unit)
			end
		end
		local function stepAccResetGrad(gradIn, gradTar, lenvec, curstep, sind, ind, bsize, rv)
			local _rv = rv or 0
			local mask = lenvec:ge(curstep)
			local bsize = bsize or gradIn[1]:size(1)
			ind:range(1, bsize)
			sind:maskedSelect(ind, mask)
			for _, unit in ipairs(gradIn) do
				gradTar[_]:indexAdd(1, sind, unit:index(1, sind))
				unit:indexFill(1, sind, _rv)
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
				if step < self.stepFull then
					stepAccResetGrad(_gradOutput, self.farGradWeight, input[2], step, self.sind, self.ind)
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
